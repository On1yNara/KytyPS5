#ifndef GRAPHICS_HOST_GPU_RENDERER_CHECKPOINTS_H_
#define GRAPHICS_HOST_GPU_RENDERER_CHECKPOINTS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

// Diagnostics only (VK_NV_device_diagnostic_checkpoints): every draw and dispatch leaves a numbered
// marker in its command buffer. After a lost device the driver reports the last marker each
// pipeline stage reached, and this table turns the number back into the shaders that ran.
namespace Libs::Graphics::Checkpoints {

struct Entry {
	uint64_t seq  = 0;
	char     kind = 0; // 'C' dispatch, 'I' indirect dispatch, 'D' draw
	uint64_t first_hash  = 0; // compute shader, or the first vertex-pipeline stage
	uint64_t second_hash = 0; // pixel shader of a draw, 0 otherwise
};

inline constexpr size_t           Capacity = size_t {1} << 16u;
inline std::array<Entry, Capacity> g_entries {};
inline std::atomic<uint64_t>       g_next {1};
inline bool                        g_enabled = false;

inline void Mark(vk::CommandBuffer command, char kind, uint64_t first_hash, uint64_t second_hash) {
	if (!g_enabled) return;
	const auto seq           = g_next.fetch_add(1, std::memory_order_relaxed);
	g_entries[seq % Capacity] = Entry {seq, kind, first_hash, second_hash};
	command.setCheckpointNV(reinterpret_cast<const void*>(static_cast<uintptr_t>(seq)));
}

inline const Entry* Find(uint64_t seq) {
	const auto& entry = g_entries[seq % Capacity];
	return entry.seq == seq ? &entry : nullptr;
}

} // namespace Libs::Graphics::Checkpoints

#endif
