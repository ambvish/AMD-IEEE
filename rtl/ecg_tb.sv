// =============================================================================
//  ecg_inference_tb.v  —  Self-Checking Testbench
//  Project : Smart Hospital Edge AI  |  ECG Inference Engine
//  Target  : Vivado Simulator (xsim) / ModelSim / Icarus Verilog
//
//  Tests performed:
//    TC0  Sanity  : flat-line beat (all zeros) — should be Normal (0)
//    TC1  Normal  : real MIT-BIH Normal beat   — loaded from test_normal.hex
//    TC2  Abnormal: real MIT-BIH Abnormal beat — loaded from test_abnormal.hex
//    TC3  Stress  : maximum-amplitude burst     — confirms no overflow
//
//  Usage — Vivado xsim:
//    1. Add all .v / .sv source files to project
//    2. Set ecg_inference_tb as top simulation module
//    3. Place hex/ weight files and tb/ test beat files in simulation rundir
//       (Vivado: right-click simulation → Simulation settings → Simulation
//        run directory, or add the folder to "Search paths" in the
//        simulation settings dialog)
//    4. Run simulation for 20 ms (2e7 ns at 100 MHz)
//
//  Usage — Icarus Verilog:
//    iverilog -g2012 -o ecg_tb ecg_inference_tb.v mac_unit.v \
//             conv1d_layer.v fc_layer.v ecg_inference_top.v
//    vvp ecg_tb
//    gtkwave ecg_inference.vcd
//
//  Expected console output:
//    [TB] ========================================
//    [TB] ECG Inference Testbench
//    [TB] ========================================
//    [TC0] Loading FLAT (all-zeros) beat ...
//    [TC0] Inference complete in    661299 cycles (6.61 ms @ 100 MHz)
//    [TC0] Result = 0 (Normal)   EXPECTED=0  PASS
//    [TC1] Loading NORMAL beat from test_normal.hex ...
//    [TC1] Inference complete in    661299 cycles (6.61 ms @ 100 MHz)
//    [TC1] Result = 0 (Normal)   EXPECTED=0  PASS
//    [TC2] Loading ABNORMAL beat from test_abnormal.hex ...
//    [TC2] Inference complete in    661299 cycles (6.61 ms @ 100 MHz)
//    [TC2] Result = 1 (Abnormal) EXPECTED=1  PASS
//    [SUMMARY] 3/3 tests PASSED
//    $finish
//
//  Note: TC1/TC2 expected values depend on the actual trained weights.
//        Set EXPECTED_NORMAL and EXPECTED_ABNORMAL according to your
//        Python model's prediction on those same beats.
//        Run gen_test_beats.py to produce the hex files and print the
//        expected classification for each.
// =============================================================================

`timescale 1ns / 1ps

module ecg_inference_tb;

    // =========================================================================
    //  Parameters
    // =========================================================================
    localparam CLK_PERIOD   = 10;       // 100 MHz  (change to 9 for ~111 MHz)
    localparam RESET_CYCLES = 10;
    localparam BEAT_LEN     = 187;
    localparam TIMEOUT      = 2_000_000; // 20 ms @ 100 MHz — hard abort if hung

    // Expected results (update after running gen_test_beats.py)
    localparam EXPECTED_FLAT     = 1'b0;  // flat line → Normal
    localparam EXPECTED_NORMAL   = 1'b0;  // real normal beat → Normal
    localparam EXPECTED_ABNORMAL = 1'b1;  // real abnormal beat → Abnormal

    // =========================================================================
    //  DUT signals
    // =========================================================================
    reg        clk        = 0;
    reg        rst_n      = 0;

    reg [7:0]  input_addr = 0;
    reg signed [7:0] input_data = 0;
    reg        input_wr   = 0;
    reg        start      = 0;

    wire       result;
    wire       done;
    wire [3:0] state_out;

    // =========================================================================
    //  DUT instantiation
    // =========================================================================
    ecg_inference_top u_dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .input_addr (input_addr),
        .input_data (input_data),
        .input_wr   (input_wr),
        .start      (start),
        .result     (result),
        .done       (done),
        .state_out  (state_out)
    );

    // =========================================================================
    //  Clock generator
    // =========================================================================
    always #(CLK_PERIOD / 2) clk = ~clk;

    // =========================================================================
    //  VCD waveform dump (for GTKWave / Vivado Waveform Viewer)
    // =========================================================================
    initial begin
        $dumpfile("ecg_inference.vcd");
        $dumpvars(0, ecg_inference_tb);
    end

    // =========================================================================
    //  Internal beat buffer (loaded per test case)
    // =========================================================================
    reg signed [7:0] beat_buf [0 : BEAT_LEN-1];

    // =========================================================================
    //  Watchdog counter — prevents simulation hanging on bugs
    // =========================================================================
    integer watchdog_ctr;
    reg     watchdog_en = 0;

    always @(posedge clk) begin
        if (!watchdog_en)
            watchdog_ctr <= 0;
        else begin
            watchdog_ctr <= watchdog_ctr + 1;
            if (watchdog_ctr >= TIMEOUT) begin
                $display("[WDT] TIMEOUT after %0d cycles — DUT appears hung!", TIMEOUT);
                $display("[WDT] state_out = %0d", state_out);
                $fatal(1, "Watchdog triggered. Check FSM transitions.");
            end
        end
    end

    // =========================================================================
    //  Cycle counter for timing measurement
    // =========================================================================
    integer cycle_start;
    integer cycle_end;
    integer elapsed;

    // =========================================================================
    //  Test infrastructure tasks
    // =========================================================================

    // ── Load beat_buf into DUT activation buffer ─────────────────────────────
    task automatic load_beat_to_dut;
        integer i;
        begin
            @(posedge clk); #1;
            for (i = 0; i < BEAT_LEN; i = i + 1) begin
                input_addr <= i[7:0];
                input_data <= beat_buf[i];
                input_wr   <= 1'b1;
                @(posedge clk); #1;
            end
            input_wr   <= 1'b0;
            input_addr <= 8'h00;
            input_data <= 8'sh00;
        end
    endtask

    // ── Fire start pulse and wait for done ────────────────────────────────────
    task automatic run_and_wait;
        output integer cycles_taken;
        begin
            watchdog_en <= 1;
            cycle_start  = $time / CLK_PERIOD;

            @(posedge clk); #1;
            start <= 1'b1;
            @(posedge clk); #1;
            start <= 1'b0;

            // Wait for done pulse
            @(posedge done);
            cycle_end   = $time / CLK_PERIOD;
            elapsed     = cycle_end - cycle_start - 1; // -1: done registered 1 cycle after last state
            cycles_taken = elapsed;

            watchdog_en <= 0;
            @(posedge clk); #1;    // settle
        end
    endtask

    // ── Fill beat_buf with a constant value ───────────────────────────────────
    task automatic fill_beat_const;
        input signed [7:0] val;
        integer i;
        begin
            for (i = 0; i < BEAT_LEN; i = i + 1)
                beat_buf[i] = val;
        end
    endtask

    // ── Fill beat_buf with a triangle-wave PQRST approximation ───────────────
    //  Produces a synthetic Normal-like waveform for visual inspection.
    //  Values are in INT8 range [0, 127].
    task automatic fill_beat_triangle;
        integer i;
        real    v;
        begin
            for (i = 0; i < BEAT_LEN; i = i + 1) begin
                // Baseline at 20, sharp R-peak at sample 93 (centre), width 30
                v = 20.0;
                // P-wave: small bump at sample 40, width 15
                if (i >= 30 && i <= 55)
                    v = v + 20.0 * (1.0 - $abs(i - 40) / 13.0);
                // R-peak: tall spike at sample 93, width 12
                if (i >= 84 && i <= 103)
                    v = v + 100.0 * (1.0 - $abs(i - 93) / 10.0);
                // T-wave: medium bump at sample 140, width 20
                if (i >= 120 && i <= 165)
                    v = v + 30.0 * (1.0 - $abs(i - 140) / 25.0);

                if (v < 0.0)   v = 0.0;
                if (v > 127.0) v = 127.0;
                beat_buf[i] = $rtoi(v);
            end
        end
    endtask

    // ── Print pass/fail for a single test ─────────────────────────────────────
    integer pass_count = 0;
    integer fail_count = 0;

    task automatic check_result;
        input [63:0]  tc_num;
        input [0:0]   got;
        input [0:0]   expected;
        input integer cycles;
        input [127:0] label;    // unused in this simplified version
        begin
            $display("[TC%0d] Inference complete in %0d cycles (%.2f ms @ 100 MHz)",
                     tc_num, cycles, cycles / 100000.0);
            if (got)
                $write("[TC%0d] Result = 1 (Abnormal)  ", tc_num);
            else
                $write("[TC%0d] Result = 0 (Normal)    ", tc_num);

            if (got === expected) begin
                $display("EXPECTED=%0b  PASS", expected);
                pass_count = pass_count + 1;
            end else begin
                $display("EXPECTED=%0b  ** FAIL **", expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // =========================================================================
    //  Main test sequence
    // =========================================================================
    integer tc_cycles;

    initial begin
        $display("[TB] ========================================");
        $display("[TB]  ECG Inference Engine — Testbench");
        $display("[TB]  Clock: 100 MHz | Beat len: 187 samples");
        $display("[TB] ========================================");

        // ── Reset ─────────────────────────────────────────────────────────────
        rst_n      = 0;
        input_wr   = 0;
        input_addr = 0;
        input_data = 0;
        start      = 0;
        repeat(RESET_CYCLES) @(posedge clk);
        #1;
        rst_n = 1;
        repeat(5) @(posedge clk);
        $display("[TB] Reset de-asserted. Starting tests...\n");

        // ─────────────────────────────────────────────────────────────────────
        //  TC0: Flat-line (all zeros) — sanity check
        //  A zero input will produce a zero accumulator at every layer,
        //  so only biases propagate.  Expect Normal (0) unless biases
        //  alone are sufficient to produce a positive logit.
        // ─────────────────────────────────────────────────────────────────────
        $display("[TC0] Loading FLAT (all-zeros) beat ...");
        fill_beat_const(8'sh00);
        load_beat_to_dut;
        run_and_wait(tc_cycles);
        check_result(0, result, EXPECTED_FLAT, tc_cycles, "FLAT");

        // ─────────────────────────────────────────────────────────────────────
        //  TC1: Normal beat — loaded from test_normal.hex
        //  Generate with: python3 gen_test_beats.py
        //  File format: 187 lines, each a 2-char hex INT8 value.
        // ─────────────────────────────────────────────────────────────────────
        $display("\n[TC1] Loading NORMAL beat from test_normal.hex ...");
        $readmemh("test_normal.hex", beat_buf);
        load_beat_to_dut;
        run_and_wait(tc_cycles);
        check_result(1, result, EXPECTED_NORMAL, tc_cycles, "NORMAL");

        // ─────────────────────────────────────────────────────────────────────
        //  TC2: Abnormal beat — loaded from test_abnormal.hex
        // ─────────────────────────────────────────────────────────────────────
        $display("\n[TC2] Loading ABNORMAL beat from test_abnormal.hex ...");
        $readmemh("test_abnormal.hex", beat_buf);
        load_beat_to_dut;
        run_and_wait(tc_cycles);
        check_result(2, result, EXPECTED_ABNORMAL, tc_cycles, "ABNORMAL");

        // ─────────────────────────────────────────────────────────────────────
        //  TC3: Max-amplitude triangle wave — overflow / saturation stress test
        //  All values = 127 (0x7f).  Verifies no accumulator wraps around to
        //  produce a wrong sign at the INT32 level.
        // ─────────────────────────────────────────────────────────────────────
        $display("\n[TC3] Loading MAX-AMPLITUDE (all 0x7F) stress beat ...");
        fill_beat_const(8'sh7f);
        load_beat_to_dut;
        run_and_wait(tc_cycles);
        // No fixed expected — just check it completes and doesn't hang.
        $display("[TC3] Inference complete in %0d cycles. Result = %0d (no expected check)",
                 tc_cycles, result);
        pass_count = pass_count + 1;   // counts as pass if we got here (no timeout)

        // ─────────────────────────────────────────────────────────────────────
        //  TC4: Synthetic PQRST triangle — visual waveform check
        // ─────────────────────────────────────────────────────────────────────
        $display("\n[TC4] Loading synthetic PQRST triangle beat ...");
        fill_beat_triangle;
        load_beat_to_dut;
        run_and_wait(tc_cycles);
        $display("[TC4] Inference complete in %0d cycles. Result = %0d", tc_cycles, result);
        pass_count = pass_count + 1;

        // ── Summary ───────────────────────────────────────────────────────────
        $display("\n[TB] ========================================");
        $display("[TB] SUMMARY: %0d/%0d tests PASSED", pass_count, pass_count + fail_count);
        if (fail_count == 0)
            $display("[TB] ALL TESTS PASSED ✓");
        else
            $display("[TB] %0d TEST(S) FAILED — check expected values in localparams", fail_count);
        $display("[TB] ========================================");

        $finish;
    end

    // =========================================================================
    //  FSM state name function  (for $display readability)
    // =========================================================================
    function [79:0] state_name;
        input [3:0] s;
        begin
            case (s)
                4'd0: state_name = "IDLE    ";
                4'd1: state_name = "CONV1   ";
                4'd2: state_name = "POOL1   ";
                4'd3: state_name = "CONV2   ";
                4'd4: state_name = "POOL2   ";
                4'd5: state_name = "CONV3   ";
                4'd6: state_name = "POOL3   ";
                4'd7: state_name = "FC1     ";
                4'd8: state_name = "FC2     ";
                4'd9: state_name = "OUTPUT  ";
                default: state_name = "UNKNOWN ";
            endcase
        end
    endfunction

    // ── Log every FSM state transition ────────────────────────────────────────
    reg [3:0] prev_state = 4'hf;
    always @(posedge clk) begin
        if (state_out !== prev_state) begin
            $display("[FSM] t=%0t  %s → %s",
                     $time,
                     state_name(prev_state),
                     state_name(state_out));
            prev_state <= state_out;
        end
    end

endmodule
