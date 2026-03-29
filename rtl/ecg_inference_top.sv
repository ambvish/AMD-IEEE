// =============================================================================
//  ecg_inference_top.v  —  Top-Level ECG Inference Engine
//  Project : Smart Hospital Edge AI  |  ECG Inference Engine
//  Target  : Zynq-7020 (Pynq-Z2 / ZedBoard / Ultra96)
//
//  Architecture:
//    Input  → Conv1(1→16,k=5,pad=2) → ReLU → MaxPool2
//           → Conv2(16→32,k=5,pad=2) → ReLU → MaxPool2
//           → Conv3(32→64,k=3,pad=1) → ReLU → MaxPool2
//           → FC1(1472→64) → ReLU
//           → FC2(64→1) → sign(logit) → classification output
//
//  Interface:
//    The PS (ARM) loads 187 INT8 samples via AXI-Lite/BRAM or directly
//    through the input_data / input_valid / input_ready handshake below.
//    Assert start after all 187 samples are loaded.
//    Read result when done=1.
//
//  Sample loading:
//    Drive input_data[7:0] and input_wr=1 for 187 consecutive cycles.
//    input_addr[7:0] selects sample index 0..186.
//
//  Timing budget @ 100 MHz (verified by RTL math check):
//    Conv1 : 16 × 187 × (1×5  +3) =  23,936 cycles
//    Pool1 : 16 × 93               =   1,488 cycles
//    Conv2 : 32 × 93  × (16×5 +3) = 247,008 cycles
//    Pool2 : 32 × 46               =   1,472 cycles
//    Conv3 : 64 × 46  × (32×3 +3) = 291,456 cycles
//    Pool3 : 64 × 23               =   1,472 cycles
//    FC1   : 64 × (1472+3)         =  94,400 cycles
//    FC2   :  1 × (64+3)           =      67 cycles
//    TOTAL :                           661,299 cycles  ≈ 6.61 ms @ 100 MHz ✓
//
//  Activation buffer sizes:
//    act0 [0:186]     187   × 8b  input
//    act1 [0:2975]   16×186 × 8b  conv1 out  (pre-pool; 16×187=2992, rounded)
//    act2 [0:1487]   16×93  × 8b  pool1 out
//    act3 [0:2975]   32×93  × 8b  conv2 out
//    act4 [0:1471]   32×46  × 8b  pool2 out
//    act5 [0:2943]   64×46  × 8b  conv3 out
//    act6 [0:1471]   64×23  × 8b  pool3 out  (= FC1 input, 1472 elements)
//    act7 [0:63]     64     × 8b  FC1  out
// =============================================================================

`include "weights_pkg.vh"

module ecg_inference_top (
    input  wire        clk,
    input  wire        rst_n,

    // ── Sample loading interface ──────────────────────────────────────────────
    input  wire [7:0]  input_addr,    // sample index 0..186
    input  wire signed [7:0] input_data, // INT8 sample value
    input  wire        input_wr,      // write strobe

    // ── Control ───────────────────────────────────────────────────────────────
    input  wire        start,         // pulse after all samples loaded

    // ── Result ────────────────────────────────────────────────────────────────
    output reg         result,        // 0 = Normal, 1 = Abnormal
    output reg         done,          // single-cycle pulse when result valid
    output reg [3:0]   state_out      // debug: current FSM state
);

    // =========================================================================
    //  Parameters (must match training pipeline exactly)
    // =========================================================================
    localparam IN_LEN     = 187;
    localparam CONV1_OLEN = 187;   // conv output length (same as input, pad compensates)
    localparam POOL1_OLEN = 93;    // 187/2 = 93 (floor)
    localparam CONV2_OLEN = 93;
    localparam POOL2_OLEN = 46;    // 93/2 = 46
    localparam CONV3_OLEN = 46;
    localparam POOL3_OLEN = 23;    // 46/2 = 23
    localparam FC1_IN     = 64 * 23;  // 1472

    // Requantisation shifts  — tune based on weights_manifest.json scales
    localparam CONV1_SHIFT = 8;
    localparam CONV2_SHIFT = 8;
    localparam CONV3_SHIFT = 8;
    localparam FC1_SHIFT   = 8;

    // =========================================================================
    //  Top-Level FSM states
    // =========================================================================
    localparam ST_IDLE      = 4'd0;
    localparam ST_CONV1     = 4'd1;
    localparam ST_POOL1     = 4'd2;
    localparam ST_CONV2     = 4'd3;
    localparam ST_POOL2     = 4'd4;
    localparam ST_CONV3     = 4'd5;
    localparam ST_POOL3     = 4'd6;
    localparam ST_FC1       = 4'd7;
    localparam ST_FC2       = 4'd8;
    localparam ST_OUTPUT    = 4'd9;

    reg [3:0] state;
    assign state_out = state;

    // =========================================================================
    //  Activation Buffers
    // =========================================================================
    reg signed [7:0] act0 [0 : IN_LEN-1];                  // input
    reg signed [7:0] act1 [0 : 16*CONV1_OLEN-1];           // conv1 out
    reg signed [7:0] act2 [0 : 16*POOL1_OLEN-1];           // pool1 out
    reg signed [7:0] act3 [0 : 32*CONV2_OLEN-1];           // conv2 out
    reg signed [7:0] act4 [0 : 32*POOL2_OLEN-1];           // pool2 out
    reg signed [7:0] act5 [0 : 64*CONV3_OLEN-1];           // conv3 out
    reg signed [7:0] act6 [0 : 64*POOL3_OLEN-1];           // pool3 out / FC1 in
    reg signed [7:0] act7 [0 : 63];                         // FC1 out

    // =========================================================================
    //  Input sample loading (PS → act0)
    // =========================================================================
    always @(posedge clk) begin
        if (input_wr)
            act0[input_addr] <= input_data;
    end

    // =========================================================================
    //  Layer start / done signals
    // =========================================================================
    reg  conv1_start, conv2_start, conv3_start;
    reg  fc1_start,   fc2_start;
    wire conv1_done,  conv2_done,  conv3_done;
    wire fc1_done,    fc2_done;

    // =========================================================================
    //  Conv1 Instance  (1→16, k=5, pad=2, IN_LEN=187)
    // =========================================================================
    conv1d_layer #(
        .IN_CH    (1),
        .OUT_CH   (16),
        .KERNEL   (5),
        .PAD      (2),
        .IN_LEN   (CONV1_OLEN),
        .OUT_LEN  (CONV1_OLEN),
        .SHIFT    (CONV1_SHIFT),
        .W_DEPTH  (16*1*5),
        .B_DEPTH  (16),
        .W_FILE   ("conv1_weights.hex"),
        .B_FILE   ("conv1_bias.hex")
    ) u_conv1 (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (conv1_start),
        .in_buf  (act0),
        .out_buf (act1),
        .done    (conv1_done)
    );

    // =========================================================================
    //  Conv2 Instance  (16→32, k=5, pad=2, IN_LEN=93)
    // =========================================================================
    conv1d_layer #(
        .IN_CH    (16),
        .OUT_CH   (32),
        .KERNEL   (5),
        .PAD      (2),
        .IN_LEN   (POOL1_OLEN),
        .OUT_LEN  (POOL1_OLEN),
        .SHIFT    (CONV2_SHIFT),
        .W_DEPTH  (32*16*5),
        .B_DEPTH  (32),
        .W_FILE   ("conv2_weights.hex"),
        .B_FILE   ("conv2_bias.hex")
    ) u_conv2 (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (conv2_start),
        .in_buf  (act2),
        .out_buf (act3),
        .done    (conv2_done)
    );

    // =========================================================================
    //  Conv3 Instance  (32→64, k=3, pad=1, IN_LEN=46)
    // =========================================================================
    conv1d_layer #(
        .IN_CH    (32),
        .OUT_CH   (64),
        .KERNEL   (3),
        .PAD      (1),
        .IN_LEN   (POOL2_OLEN),
        .OUT_LEN  (POOL2_OLEN),
        .SHIFT    (CONV3_SHIFT),
        .W_DEPTH  (64*32*3),
        .B_DEPTH  (64),
        .W_FILE   ("conv3_weights.hex"),
        .B_FILE   ("conv3_bias.hex")
    ) u_conv3 (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (conv3_start),
        .in_buf  (act4),
        .out_buf (act5),
        .done    (conv3_done)
    );

    // =========================================================================
    //  FC1 Instance  (1472→64, with ReLU)
    // =========================================================================
    fc_layer #(
        .IN_SIZE    (FC1_IN),
        .OUT_SIZE   (64),
        .SHIFT      (FC1_SHIFT),
        .APPLY_RELU (1),
        .W_DEPTH    (64 * FC1_IN),
        .B_DEPTH    (64),
        .W_FILE     ("fc1_weights.hex"),
        .B_FILE     ("fc1_bias.hex")
    ) u_fc1 (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (fc1_start),
        .in_vec  (act6),
        .out_vec (act7),
        .out_raw (/* unused */),
        .done    (fc1_done)
    );

    // =========================================================================
    //  FC2 Instance  (64→1, no ReLU — raw logit for sign comparison)
    // =========================================================================
    wire signed [7:0]  fc2_out_vec [0:0];
    wire signed [31:0] fc2_out_raw;

    fc_layer #(
        .IN_SIZE    (64),
        .OUT_SIZE   (1),
        .SHIFT      (8),
        .APPLY_RELU (0),
        .W_DEPTH    (1 * 64),
        .B_DEPTH    (1),
        .W_FILE     ("fc2_weights.hex"),
        .B_FILE     ("fc2_bias.hex")
    ) u_fc2 (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (fc2_start),
        .in_vec  (act7),
        .out_vec (fc2_out_vec),
        .out_raw (fc2_out_raw),
        .done    (fc2_done)
    );

    // =========================================================================
    //  MaxPool helper: runs as part of the top FSM using counters
    // =========================================================================
    reg [7:0]  pool_ch;      // channel index
    reg [7:0]  pool_t;       // output time index (= input_time / 2)
    reg [3:0]  pool_state;   // which pool stage we're in

    // Inline max-of-two
    function signed [7:0] max8;
        input signed [7:0] a, b;
        begin
            max8 = (a > b) ? a : b;
        end
    endfunction

    // =========================================================================
    //  Top FSM
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            done        <= 1'b0;
            result      <= 1'b0;
            conv1_start <= 1'b0;
            conv2_start <= 1'b0;
            conv3_start <= 1'b0;
            fc1_start   <= 1'b0;
            fc2_start   <= 1'b0;
            pool_ch     <= 0;
            pool_t      <= 0;
        end else begin
            // Default: deassert single-cycle start pulses
            conv1_start <= 1'b0;
            conv2_start <= 1'b0;
            conv3_start <= 1'b0;
            fc1_start   <= 1'b0;
            fc2_start   <= 1'b0;
            done        <= 1'b0;

            case (state)

                // ─────────────────────────────────────────────────────────────
                ST_IDLE: begin
                    if (start) begin
                        conv1_start <= 1'b1;
                        state       <= ST_CONV1;
                    end
                end

                // ─────────────────────────────────────────────────────────────
                ST_CONV1: begin
                    if (conv1_done) begin
                        // Begin MaxPool1: collapse act1 (16×187) → act2 (16×93)
                        pool_ch <= 0;
                        pool_t  <= 0;
                        state   <= ST_POOL1;
                    end
                end

                // ── MaxPool1: act1 → act2 ────────────────────────────────────
                //  act1 layout: [oc * CONV1_OLEN + t]
                //  act2 layout: [oc * POOL1_OLEN + t_pool]
                ST_POOL1: begin
                    begin : pool1_block
                        reg signed [7:0] v0, v1;
                        v0 = act1[pool_ch * CONV1_OLEN + pool_t * 2];
                        v1 = act1[pool_ch * CONV1_OLEN + pool_t * 2 + 1];
                        act2[pool_ch * POOL1_OLEN + pool_t] <= max8(v0, v1);
                    end

                    if (pool_t == POOL1_OLEN - 1) begin
                        pool_t <= 0;
                        if (pool_ch == 15) begin       // 16 channels
                            conv2_start <= 1'b1;
                            state       <= ST_CONV2;
                        end else begin
                            pool_ch <= pool_ch + 1;
                        end
                    end else begin
                        pool_t <= pool_t + 1;
                    end
                end

                // ─────────────────────────────────────────────────────────────
                ST_CONV2: begin
                    if (conv2_done) begin
                        pool_ch <= 0;
                        pool_t  <= 0;
                        state   <= ST_POOL2;
                    end
                end

                // ── MaxPool2: act3 → act4 ────────────────────────────────────
                ST_POOL2: begin
                    begin : pool2_block
                        reg signed [7:0] v0, v1;
                        v0 = act3[pool_ch * CONV2_OLEN + pool_t * 2];
                        v1 = act3[pool_ch * CONV2_OLEN + pool_t * 2 + 1];
                        act4[pool_ch * POOL2_OLEN + pool_t] <= max8(v0, v1);
                    end

                    if (pool_t == POOL2_OLEN - 1) begin
                        pool_t <= 0;
                        if (pool_ch == 31) begin       // 32 channels
                            conv3_start <= 1'b1;
                            state       <= ST_CONV3;
                        end else begin
                            pool_ch <= pool_ch + 1;
                        end
                    end else begin
                        pool_t <= pool_t + 1;
                    end
                end

                // ─────────────────────────────────────────────────────────────
                ST_CONV3: begin
                    if (conv3_done) begin
                        pool_ch <= 0;
                        pool_t  <= 0;
                        state   <= ST_POOL3;
                    end
                end

                // ── MaxPool3: act5 → act6  (flattened; FC1 reads act6) ───────
                //  act5 layout: [oc * CONV3_OLEN + t]
                //  act6 layout: [oc * POOL3_OLEN + t]  (= FC1 linear input)
                ST_POOL3: begin
                    begin : pool3_block
                        reg signed [7:0] v0, v1;
                        v0 = act5[pool_ch * CONV3_OLEN + pool_t * 2];
                        v1 = act5[pool_ch * CONV3_OLEN + pool_t * 2 + 1];
                        act6[pool_ch * POOL3_OLEN + pool_t] <= max8(v0, v1);
                    end

                    if (pool_t == POOL3_OLEN - 1) begin
                        pool_t <= 0;
                        if (pool_ch == 63) begin       // 64 channels
                            fc1_start <= 1'b1;
                            state     <= ST_FC1;
                        end else begin
                            pool_ch <= pool_ch + 1;
                        end
                    end else begin
                        pool_t <= pool_t + 1;
                    end
                end

                // ─────────────────────────────────────────────────────────────
                ST_FC1: begin
                    if (fc1_done) begin
                        fc2_start <= 1'b1;
                        state     <= ST_FC2;
                    end
                end

                // ─────────────────────────────────────────────────────────────
                ST_FC2: begin
                    if (fc2_done)
                        state <= ST_OUTPUT;
                end

                // ── Classification: sign of raw INT32 logit ─────────────────
                //  sigmoid(logit) > 0.5  ↔  logit > 0  → Abnormal (1)
                //  sigmoid(logit) ≤ 0.5  ↔  logit ≤ 0  → Normal   (0)
                //  Single comparison on the full INT32 value — no scaling needed.
                ST_OUTPUT: begin
                    result <= ($signed(fc2_out_raw) > 32'sd0) ? 1'b1 : 1'b0;
                    done  <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
