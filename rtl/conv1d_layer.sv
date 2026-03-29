// =============================================================================
//  mac_unit.v  —  Registered Multiply-Accumulate Unit
//  Project : Smart Hospital Edge AI  |  ECG Inference Engine
//  Target  : Zynq-7020
//
//  Operation  (all registered on posedge clk):
//    clear=1  →  acc ← bias_in            (preload bias)
//    en=1     →  acc ← acc + weight * act  (accumulate)
//    else     →  acc holds
//
//  Arithmetic:
//    weight : signed  8-bit  (INT8, two's complement)
//    act    : signed  8-bit  (INT8, two's complement)
//    product: signed 16-bit  (sign-extended to 32 before add)
//    acc    : signed 32-bit  (INT32, never overflows for ≤2^23 terms)
//
//  Latency: 1 clock cycle  (registered output)
//    → caller must read acc ONE cycle after the final en pulse.
// =============================================================================

module mac_unit (
    input  wire        clk,
    input  wire        rst_n,

    input  wire        clear,            // 1: load bias_in into acc (synchronous)
    input  wire        en,               // 1: accumulate weight*act into acc

    input  wire signed [7:0]  weight,   // INT8 filter coefficient
    input  wire signed [7:0]  act,      // INT8 activation (input to this layer)
    input  wire signed [31:0] bias_in,  // INT32 bias (preloaded on clear)

    output reg  signed [31:0] acc       // INT32 running accumulator
);

    // ------------------------------------------------------------------
    //  16-bit product, sign-extended to 32 bits before accumulation.
    //  $signed ensures arithmetic (not logical) multiplication.
    // ------------------------------------------------------------------
    wire signed [15:0] product_16 = $signed(weight) * $signed(act);
    wire signed [31:0] product_32 = {{16{product_16[15]}}, product_16};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 32'sd0;
        else if (clear)
            acc <= bias_in;             // preload: bias already quantised (INT32)
        else if (en)
            acc <= acc + product_32;    // saturating add not needed: max terms = 6144
    end

endmodule
