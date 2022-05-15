import os
import KpiScripts.CompileNetworks.Kpi_compile as kpi_compile
import KpiScripts.SingleImgComparison.singleImgComparison as single_comparison

if __name__ == "__main__":
    ov_path = os.sys.argv[1]
    print("ov_path: ", ov_path)
    # execute compiling unis_insight network xmls
    kpi_compile.main(ov_path)
    # execute cross check unis_insight networl blobs
    single_comparison.main(ov_path)
