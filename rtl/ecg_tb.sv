module ecg_tb;

    reg clk = 0;
    reg rst_n = 0;

    wire out;

    always #5 clk = ~clk;

    ecg_inference_top dut (
        .clk(clk),
        .rst_n(rst_n),
        .out(out)
    );

    initial begin
        $display("Starting simulation...");

        rst_n = 0;
        #20;
        rst_n = 1;

        #1000000;

        $display("Output = %d", out);

        $finish;
    end

endmodule
