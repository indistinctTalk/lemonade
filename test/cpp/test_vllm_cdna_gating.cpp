#include <iostream>
#include <string>

#include "lemon/backends/vllm/vllm_arg_resolver.h"
#include "lemon/system_info.h"

using lemon::SystemInfo;
using lemon::backends::device_class_launch_policy;
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

    // Per-arch version override: gfx942 (CDNA-dcgpu) pins a distinct vLLM/ROCm
    // release line from the RDNA default, since no single tag carries both.
    expect(SystemInfo::vllm_rocm_version_override("gfx942") == "vllm0.19.1-rocm7.13.0",
           "vllm gfx942 overrides to its own dcgpu release line");
    expect(SystemInfo::vllm_rocm_version_override("gfx110X").empty(),
           "vllm RDNA families use the default pin (no override)");
    expect(SystemInfo::vllm_rocm_version_override("gfx1151").empty(),
           "vllm gfx1151 uses the default pin (no override)");

    // Device-class launch policy — the discrete-HBM vs conservative-default wiring
    // that VLLMServer::load() applies (extracted so it is unit-testable off-GPU).
    auto apu = device_class_launch_policy("gfx1151", false);
    expect(apu.enforce_eager && apu.cap_kv_cache && apu.force_awq_kernel,
           "gfx1151 (Strix Halo APU) keeps conservative defaults: eager + kv-cap + awq-force");
    auto cdna = device_class_launch_policy("gfx942", false);
    expect(!cdna.enforce_eager && !cdna.cap_kv_cache && !cdna.force_awq_kernel,
           "gfx942 (MI300X) gets vLLM-native budgeting: no eager, no kv-cap, no awq-force");
    auto cdna_budget = device_class_launch_policy("gfx942", true);
    expect(!cdna_budget.cap_kv_cache,
           "explicit user memory budget suppresses the kv-cap on discrete-HBM");
    auto rdna_dgpu = device_class_launch_policy("gfx1100", false);
    expect(rdna_dgpu.enforce_eager && rdna_dgpu.cap_kv_cache && rdna_dgpu.force_awq_kernel,
           "gfx1100 (consumer dGPU) keeps conservative defaults");

    // Escape hatch: an explicit --enforce-eager forces eager even on discrete-HBM
    // (for a model whose CUDA-graph capture misbehaves), without disturbing the
    // other discrete-HBM defaults.
    auto cdna_eager = device_class_launch_policy("gfx942", false, /*has_enforce_eager=*/true);
    expect(cdna_eager.enforce_eager,
           "explicit --enforce-eager overrides the discrete-HBM graph default on gfx942");
    expect(!cdna_eager.cap_kv_cache && !cdna_eager.force_awq_kernel,
           "the eager escape hatch does not disturb the other gfx942 defaults");

    if (failures != 0) {
        std::cout << failures << " assertion(s) failed" << std::endl;
        return 1;
    }
    std::cout << "All vllm CDNA gating assertions passed" << std::endl;
    return 0;
}
