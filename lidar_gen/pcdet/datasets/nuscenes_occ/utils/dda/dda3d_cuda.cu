#include <cuda_runtime.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)

__global__ void dda3d_kernel(
    const float* rays, const bool* grid, bool* intersections,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    // 获取当前射线的方向
    float dx = rays[i * 3 + 0];
    float dy = rays[i * 3 + 1];
    float dz = rays[i * 3 + 2];

    // 计算射线起点在占用网格中的索引
    int x = (0 - Xmin) / vx;
    int y = (0 - Ymin) / vy;
    int z = (0 - Zmin) / vz;

    int stepX = (dx > 0) ? 1 : -1;
    int stepY = (dy > 0) ? 1 : -1;
    int stepZ = (dz > 0) ? 1 : -1;

    float tDeltaX = (dx != 0) ? abs(vx / dx) : INFINITY;
    float tDeltaY = (dy != 0) ? abs(vy / dy) : INFINITY;
    float tDeltaZ = (dz != 0) ? abs(vz / dz) : INFINITY;

    float tMaxX = (dx > 0) ? (Xmin + (x + 1) * vx) / dx : (Xmin + x * vx) / dx;
    float tMaxY = (dy > 0) ? (Ymin + (y + 1) * vy) / dy : (Ymin + y * vy) / dy;
    float tMaxZ = (dz > 0) ? (Zmin + (z + 1) * vz) / dz : (Zmin + z * vz) / dz;

    // 遍历射线经过的体素
    while ((0 <= x && x < W) && (0 <= y && y < H) && (0 <= z && z < D)) {
        // 注意，grid大小为[W, H, D]
        int grid_index = z + y * D + x * H * D;

        // 如果体素被占用，设置结果为 true
        if (grid[grid_index]) {
            intersections[i] = true;
            return;
        }

        // 更新步进逻辑，选择最小的 tMax 并更新相应的索引
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }
    intersections[i] = false;  // 如果没有相交，结果为 false
}


__global__ void raycast_kernel(
    const float* rays, const bool* grid, bool* intersections, int* hits,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rays) return;

    // 获取当前射线的方向
    float dx = rays[i * 3 + 0];
    float dy = rays[i * 3 + 1];
    float dz = rays[i * 3 + 2];

    // 计算射线起点在占用网格中的索引
    int x = (0 - Xmin) / vx;
    int y = (0 - Ymin) / vy;
    int z = (0 - Zmin) / vz;

    int stepX = (dx > 0) ? 1 : -1;
    int stepY = (dy > 0) ? 1 : -1;
    int stepZ = (dz > 0) ? 1 : -1;

    float tDeltaX = (dx != 0) ? abs(vx / dx) : INFINITY;
    float tDeltaY = (dy != 0) ? abs(vy / dy) : INFINITY;
    float tDeltaZ = (dz != 0) ? abs(vz / dz) : INFINITY;

    float tMaxX = (dx > 0) ? (Xmin + (x + 1) * vx) / dx : (Xmin + x * vx) / dx;
    float tMaxY = (dy > 0) ? (Ymin + (y + 1) * vy) / dy : (Ymin + y * vy) / dy;
    float tMaxZ = (dz > 0) ? (Zmin + (z + 1) * vz) / dz : (Zmin + z * vz) / dz;

    // 遍历射线经过的体素
    while ((0 <= x && x < W) && (0 <= y && y < H) && (0 <= z && z < D)) {
        // 注意，grid大小为[W, H, D]
        int grid_index = z + y * D + x * H * D;

        // 如果体素被占用，设置结果为 true
        if (grid[grid_index]) {
            intersections[i] = true;
            hits[i] = x;
            hits[i + 1] = y;
            hits[i + 2] = z;
            return;
        }

        // 更新步进逻辑，选择最小的 tMax 并更新相应的索引
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                x += stepX;
                tMaxX += tDeltaX;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                y += stepY;
                tMaxY += tDeltaY;
            } else {
                z += stepZ;
                tMaxZ += tDeltaZ;
            }
        }
    }
    intersections[i] = false;  // 如果没有相交，结果为 false
}

void dda3d_launcher(const float* rays, const bool* grid, bool* intersections,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays){
    dim3 blockSize(DIVUP(num_rays, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    dda3d_kernel<<<blockSize, threadSize>>>(rays, grid, intersections, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D, num_rays);
}

void raycast_launcher(const float* rays, const bool* grid, bool* intersections, int* hits,
    float Xmin, float Ymin, float Zmin, float vx, float vy, float vz,
    int W, int H, int D, int num_rays){
    dim3 blockSize(DIVUP(num_rays, THREADS_PER_BLOCK));
    dim3 threadSize(THREADS_PER_BLOCK);
    raycast_kernel<<<blockSize, threadSize>>>(rays, grid, intersections, hits, Xmin, Ymin, Zmin, vx, vy, vz, W, H, D, num_rays);
}