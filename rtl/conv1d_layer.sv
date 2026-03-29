// =============================================================================
//  conv1d_layer.v  —  Sequential Parameterised 1D Convolution Layer
//  Project : Smart Hospital Edge AI  |  ECG Inference Engine
//  Target  : Zynq-7020
//
//  Computes one output element (oc, t) per (IN_CH * KERNEL) + 2 clock cycles:
//    - 1 cycle  : preload bias  (CLEAR state)
//    - IN_CH*K  : multiply-accumulate loop
//    - 1 cycle  : wait for final MAC to register (LATCH state)
//    - 1 cycle  : ReLU + requantise + write output (WRITE state)
//
//  Weight ROM layout (from weight_extractor.py):
//    addr = oc * (IN_CH * KERNEL) + ic * KERNEL + k
//    Total words = OUT_CH * IN_CH * KERNEL
//
//  Requantisation  (per-tensor approximation for hackathon):
//    out_int8 = clip( acc[ACC_W-1 : SHIFT] , 0, 127 )   after ReLU
//    SHIFT parameter must be tuned post-training (default 8).
//    See weights_manifest.json for per-channel scales if finer control needed.
//
//  Interface:
//    Caller provides a flat input buffer (in_buf) and receives a flat
//    output buffer (out_buf), both with word-width DATA_W bits.
//    Buffer is written incrementally as computation proceeds.
//    done goes high for one cycle when all OUT_CH*OUT_LEN elements are written.
// =============================================================================

module conv1d_layer #(
    // ── Network dimensions ────────────────────────────────────────────────────
    parameter IN_CH     = 1,     // input  channels
    parameter OUT_CH    = 16,    // output channels (filters)
    parameter KERNEL    = 5,     // kernel size (must be odd for symmetric pad)
    parameter PAD       = 2,     // zero-padding each side
    parameter IN_LEN    = 187,   // input  time length  (after padding handled here)
    parameter OUT_LEN   = 187,   // output time length  = IN_LEN + 2*PAD - KERNEL + 1
    // ── Quantisation ─────────────────────────────────────────────────────────
    parameter SHIFT     = 8,     // arithmetic right-shift for requantisation
    parameter DATA_W    = 8,     // weight / activation bit width
    parameter ACC_W     = 32,    // accumulator bit width
    // ── ROM parameters ────────────────────────────────────────────────────────
    parameter W_DEPTH   = OUT_CH * IN_CH * KERNEL,  // weight ROM depth
    parameter B_DEPTH   = OUT_CH,                    // bias ROM depth
    parameter W_FILE    = "conv1_weights.hex",
    parameter B_FILE    = "conv1_bias.hex"
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,            // pulse high for 1 cycle to begin

    // ── Input activation buffer (flat: ic*IN_LEN + t) ────────────────────────
    input  wire signed [DATA_W-1:0] in_buf  [0 : IN_CH*IN_LEN-1],

    // ── Output activation buffer (flat: oc*OUT_LEN + t) ─────────────────────
    output reg  signed [DATA_W-1:0] out_buf [0 : OUT_CH*OUT_LEN-1],

    output reg  done                     // single-cycle pulse when complete
);

    // =========================================================================
    //  Weight and Bias ROMs
    // =========================================================================
    reg signed [DATA_W-1:0]  w_rom [0 : W_DEPTH-1];
    reg signed [ACC_W-1:0]   b_rom [0 : B_DEPTH-1];

    initial begin
        $readmemh(W_FILE, w_rom);
        $readmemh(B_FILE, b_rom);
    end

    // =========================================================================
    //  State machine
    // =========================================================================
    localparam S_IDLE   = 3'd0;
    localparam S_CLEAR  = 3'd1;   // preload accumulator with bias[oc]
    localparam S_MAC    = 3'd2;   // inner loop: ic × k iterations
    localparam S_LATCH  = 3'd3;   // wait 1 cycle for last MAC result to register
    localparam S_WRITE  = 3'd4;   // ReLU + shift + write to out_buf
    localparam S_DONE   = 3'd5;

    reg [2:0]  state;

    // ── Loop counters ─────────────────────────────────────────────────────────
    //  Widths: clog2-sized.  Using 16-bit is safe for all layer sizes here.
    reg [7:0]  oc;   // current output channel  [0 .. OUT_CH-1]
    reg [7:0]  t;    // current output time step [0 .. OUT_LEN-1]
    reg [7:0]  ic;   // inner: input channel     [0 .. IN_CH-1]
    reg [3:0]  k;    // inner: kernel position   [0 .. KERNEL-1]

    // =========================================================================
    //  MAC unit wiring
    // =========================================================================
    reg                  mac_clear;
    reg                  mac_en;
    reg  signed [DATA_W-1:0] mac_weight;
    reg  signed [DATA_W-1:0] mac_act;
    wire signed [ACC_W-1:0]  mac_acc;

    mac_unit u_mac (
        .clk      (clk),
        .rst_n    (rst_n),
        .clear    (mac_clear),
        .en       (mac_en),
        .weight   (mac_weight),
        .act      (mac_act),
        .bias_in  (b_rom[oc]),
        .acc      (mac_acc)
    );

    // =========================================================================
    //  Weight + activation lookup (combinational, feeds MAC on same cycle)
    // =========================================================================

    // Weight ROM address for current (oc, ic, k)
    wire [31:0] w_addr = oc * (IN_CH * KERNEL) + ic * KERNEL + k;

    // Input position in the original (un-padded) time axis
    wire signed [15:0] in_pos = $signed({8'b0, t}) + $signed({8'b0, k})
                                 - $signed({{12{1'b0}}, PAD[3:0]});

    // Activation: zero if outside valid range (implements zero-padding)
    wire in_valid = (in_pos >= 0) && (in_pos < IN_LEN);
    wire [31:0] in_addr  = ic * IN_LEN + in_pos[7:0];  // safe: in_valid checked

    // =========================================================================
    //  FSM
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            oc         <= 0;
            t          <= 0;
            ic         <= 0;
            k          <= 0;
            done       <= 1'b0;
            mac_clear  <= 1'b0;
            mac_en     <= 1'b0;
            mac_weight <= 0;
            mac_act    <= 0;
        end else begin
            // Default: deassert single-cycle signals
            mac_clear <= 1'b0;
            mac_en    <= 1'b0;
            done      <= 1'b0;

            case (state)

                // ── Wait for start pulse ─────────────────────────────────────
                S_IDLE: begin
                    if (start) begin
                        oc    <= 0;
                        t     <= 0;
                        ic    <= 0;
                        k     <= 0;
                        state <= S_CLEAR;
                    end
                end

                // ── Preload accumulator with bias[oc] ────────────────────────
                //  mac_unit registers bias on the NEXT posedge, so MAC starts
                //  the cycle after CLEAR.
                S_CLEAR: begin
                    mac_clear <= 1'b1;       // acc ← bias_in (registered next cycle)
                    ic        <= 0;
                    k         <= 0;
                    state     <= S_MAC;
                end

                // ── Inner loop: accumulate over (ic, k) ──────────────────────
                S_MAC: begin
                    mac_en     <= in_valid;  // zero-pad: don't accumulate if out-of-bounds
                    mac_weight <= w_rom[w_addr];
                    mac_act    <= in_valid ? in_buf[in_addr] : 8'sd0;

                    // Advance k
                    if (k == KERNEL - 1) begin
                        k <= 0;
                        if (ic == IN_CH - 1) begin
                            // All (ic,k) done for this (oc,t) → wait for last MAC
                            ic    <= 0;
                            state <= S_LATCH;
                        end else begin
                            ic <= ic + 1;
                        end
                    end else begin
                        k <= k + 1;
                    end
                end

                // ── Wait 1 cycle for final MAC result to register ─────────────
                S_LATCH: begin
                    state <= S_WRITE;
                end

                // ── ReLU + arithmetic right-shift + write output ──────────────
                S_WRITE: begin
                    begin : relu_requant
                        reg signed [ACC_W-1:0] shifted;
                        reg signed [DATA_W-1:0] clamped;

                        shifted = mac_acc >>> SHIFT;  // arithmetic right shift

                        // ReLU: clip to 0 if negative; clip to 127 if overflow
                        if (shifted[ACC_W-1])           // sign bit set → negative
                            clamped = 8'sd0;
                        else if (shifted > 127)
                            clamped = 8'sd127;
                        else
                            clamped = shifted[DATA_W-1:0];

                        out_buf[oc * OUT_LEN + t] <= clamped;
                    end

                    // Advance (oc, t) counters
                    if (t == OUT_LEN - 1) begin
                        t <= 0;
                        if (oc == OUT_CH - 1) begin
                            state <= S_DONE;
                        end else begin
                            oc    <= oc + 1;
                            state <= S_CLEAR;   // load next filter's bias
                        end
                    end else begin
                        t     <= t + 1;
                        state <= S_CLEAR;       // load same filter's bias for next t
                    end
                end

                // ── Signal completion ─────────────────────────────────────────
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
