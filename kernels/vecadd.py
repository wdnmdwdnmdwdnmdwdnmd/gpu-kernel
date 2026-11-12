import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(A, B, C,
               N: tl.constexpr,
               BLOCK: tl.constexpr):

    pid = tl.program_id(0)

    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    mask = offsets < N

    a = tl.load(A + offsets, mask=mask, other=0)
    b = tl.load(B + offsets, mask=mask, other=0)

    result = a + b

    tl.store(C + offsets, result, mask=mask)


if __name__ == "__main__":

    N = 1000
    BLOCK = 256

    a = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")
    c = torch.empty_like(a)

    grid = (triton.cdiv(N, BLOCK),)

    kernel = add_kernel[grid](
        a, b, c,
        N,
        BLOCK=BLOCK,
        num_warps=4
    )

    torch.cuda.synchronize()

    expected = a + b
    torch.testing.assert_close(c, expected)

    print("Correctness: PASS")
    print("Number of programs:", grid[0])

    # 获取编译后的资源信息
    print("Registers per thread:", kernel.n_regs)

    # 导出 PTX
    ptx = kernel.asm["ptx"]

    with open("vector_add.ptx", "w") as f:
        f.write(ptx)

    print("PTX saved to vector_add.ptx")