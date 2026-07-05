#include <iostream>
#include <string>

#include "lemon/backends/vllm/vllm_arg_resolver.h"
#include "lemon/system_info.h"

using lemon::SystemInfo;
using lemon::backends::is_discrete_hbm_arch;

namespace {

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) {
        std::cout << "PASS: " << label << std::endl;
    } else {
        std::cout << "FAIL: " << label << std::endl;
        ++failures;
    }
}

}  // namespace

int main() {
    expect(SystemInfo::backend_supports_arch("vllm", "rocm", "gfx942"),
           "vllm:rocm supports gfx942 (MI300X)");

    expect(SystemInfo::backend_supports_arch("vllm", "rocm", "gfx1100"),
           "vllm:rocm still supports gfx1100 via gfx110X wildcard");
    expect(SystemInfo::backend_supports_arch("vllm", "rocm", "gfx1151"),
           "vllm:rocm still supports gfx1151 (Strix Halo)");

    expect(!SystemInfo::backend_supports_arch("vllm", "rocm", "gfx906"),
           "vllm:rocm does not support gfx906 (not published)");
    expect(!SystemInfo::backend_supports_arch("vllm", "rocm", "gfx1036"),
           "vllm:rocm does not support gfx1036 (iGPU, not published)");

    expect(is_discrete_hbm_arch("gfx942"), "gfx942 is discrete-HBM class");
    expect(is_discrete_hbm_arch("gfx950"), "gfx950 is discrete-HBM class");
    expect(is_discrete_hbm_arch("gfx90a"), "gfx90a is discrete-HBM class");
    expect(!is_discrete_hbm_arch("gfx1151"), "gfx1151 (Strix Halo APU) is not discrete-HBM");
    expect(!is_discrete_hbm_arch("gfx1100"), "gfx1100 (consumer dGPU) keeps conservative defaults");
    expect(!is_discrete_hbm_arch("gfx1201"), "gfx1201 (RDNA4) keeps conservative defaults");
    expect(!is_discrete_hbm_arch(""), "empty arch is not discrete-HBM");

    expect(SystemInfo::rocm_asset_family("gfx942") == "gfx942",
           "rocm_asset_family passes gfx942 through unchanged (release tag arch)");
    expect(SystemInfo::rocm_asset_family("gfx1100") == "gfx110X",
           "rocm_asset_family collapses gfx1100 to gfx110X");

    if (failures != 0) {
        std::cout << failures << " assertion(s) failed" << std::endl;
        return 1;
    }
    std::cout << "All vllm CDNA gating assertions passed" << std::endl;
    return 0;
}
