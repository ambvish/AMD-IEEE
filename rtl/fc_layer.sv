// =============================================================================
//  fc_layer.v  —  Sequential Fully-Connected (Linear) Layer
//  Project : Smart Hospital Edge AI  |  ECG Inference Engine
//  Target  : Zynq-7020
//
//  Computes: out[row] = ReLU( Σ_col W[row,col] * in[col] + bias[row] )
//            (ReLU is optional via APPLY_RELU parameter)
//
//  Weight ROM layout (from weight_extractor.py):
//    addr = row * IN_SIZE + col
//    Total words = OUT_SIZE * IN_SIZE
//
//  Timing per output neuron:
//    1 cycle  (CLEAR)  : preload bias
//    IN_SIZE  (MAC)    : accumulate all inputs
//    1 cycle  (LATCH)  : pipeline drain
//    1 cycle  (WRITE)  : requantise + write
//
//  Total cycles ≈ OUT_SIZE × (IN_SIZE + 3)
//    FC1: 64 × (1472+3) = 94,400 cycles  ≈ 0.94 ms @ 100 MHz
//    FC2: 1  × (64+3)   =     67 cycles  < 1 µs
// =============================================================================

module fc_layer #(
    parameter IN_SIZE   = 1472,   // input vector length (flattened conv output)
    parameter OUT_SIZE  = 64,     // output vector length
    parameter SHIFT     = 8,      // requantisation arithmetic right-shift
    parameter APPLY_RELU = 1,     // 1 = apply ReLU, 0 = pass raw value (FC2)
    parameter DATA_W    = 8,
    parameter ACC_W     = 32,
    parameter W_DEPTH   = OUT_SIZE * IN_SIZE,
    parameter B_DEPTH   = OUT_SIZE,
    parameter W_FILE    = "fc1_weights.hex",
    parameter B_FILE    = "fc1_bias.hex"
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,

    // ── Input vector ─────────────────────────────────────────────────────────
    input  wire signed [DATA_W-1:0] in_vec  [0 : IN_SIZE-1],

    // ── Output vector ────────────────────────────────────────────────────────
    //  APPLY_RELU=1  → INT8  clipped to [0,127]
    //  APPLY_RELU=0  → raw INT32 acc (only lower DATA_W bits populated on write
    //                  unless caller uses out_raw)
    output reg  signed [DATA_W-1:0] out_vec [0 : OUT_SIZE-1],
    output reg  signed [ACC_W-1:0]  out_raw,   // raw INT32 accumulator (for FC2)

    output reg  done
);

    // =========================================================================
    //  Weight and Bias ROMs
    // =========================================================================
    reg signed [DATA_W-1:0] w_rom [0 : W_DEPTH-1];
    reg signed [ACC_W-1:0]  b_rom [0 : B_DEPTH-1];

    initial begin
        $readmemh(W_FILE, w_rom);
        $readmemh(B_FILE, b_rom);
    end

    // =========================================================================
    //  State machine
    // =========================================================================
    localparam S_IDLE  = 3'd0;
    localparam S_CLEAR = 3'd1;
    localparam S_MAC   = 3'd2;
    localparam S_LATCH = 3'd3;
    localparam S_WRITE = 3'd4;
    localparam S_DONE  = 3'd5;

    reg [2:0]  state;
    reg [9:0]  row;    // output neuron index  [0 .. OUT_SIZE-1]  (10-bit: max 1024)
    reg [17:0] col;    // input index          [0 .. IN_SIZE-1]   (18-bit: max 262143)

    // =========================================================================
    //  MAC unit instance
    // =========================================================================
    reg                   mac_clear;
    reg                   mac_en;
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
        .bias_in  (b_rom[row]),
        .acc      (mac_acc)
    );

    // Weight ROM address: row × IN_SIZE + col
    wire [31:0] w_addr = row * IN_SIZE + col;

    // =========================================================================
    //  FSM
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            row        <= 0;
            col        <= 0;
            done       <= 1'b0;
            out_raw    <= 32'sd0;
            mac_clear  <= 1'b0;
            mac_en     <= 1'b0;
            mac_weight <= 0;
            mac_act    <= 0;
        end else begin
            mac_clear <= 1'b0;
            mac_en    <= 1'b0;
            done      <= 1'b0;

            case (state)

                S_IDLE: begin
                    if (start) begin
                        row   <= 0;
                        col   <= 0;
                        state <= S_CLEAR;
                    end
                end

                // ── Preload accumulator with bias[row] ───────────────────────
                S_CLEAR: begin
                    mac_clear <= 1'b1;
                    col       <= 0;
                    state     <= S_MAC;
                end

                // ── MAC over all input elements for this neuron ──────────────
                S_MAC: begin
                    mac_en     <= 1'b1;
                    mac_weight <= w_rom[w_addr];
                    mac_act    <= in_vec[col];

                    if (col == IN_SIZE - 1) begin
                        col   <= 0;
                        state <= S_LATCH;
                    end else begin
                        col <= col + 1;
                    end
                end

                // ── Drain pipeline (1 cycle) ─────────────────────────────────
                S_LATCH: begin
                    state <= S_WRITE;
                end

                // ── Requantise and write ──────────────────────────────────────
                S_WRITE: begin
                    begin : write_block
                        reg signed [ACC_W-1:0] shifted;

                        out_raw <= mac_acc;      // always expose raw value

                        if (APPLY_RELU) begin
                            shifted = mac_acc >>> SHIFT;
                            if (shifted[ACC_W-1])
                                out_vec[row] <= 8'sd0;
                            else if (shifted > 127)
                                out_vec[row] <= 8'sd127;
                            else
                                out_vec[row] <= shifted[DATA_W-1:0];
                        end else begin
                            // FC2: no ReLU, store sign-extended lower byte for
                            // threshold check; caller uses out_raw for full precision
                            out_vec[row] <= mac_acc[DATA_W-1:0];
                        end
                    end

                    if (row == OUT_SIZE - 1)
                        state <= S_DONE;
                    else begin
                        row   <= row + 1;
                        state <= S_CLEAR;
                    end
                end

                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
