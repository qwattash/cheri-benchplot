from enum import Enum


class BenchmarkCPU(Enum):
    FLUTE = "flute"
    TOOOBA = "toooba"
    MORELLO = "morello"
    BERI = "beri"
    QEMU_RISCV = "qemu-riscv"
    QEMU_MORELLO = "qemu-morello"

    def __str__(self):
        return self.value
