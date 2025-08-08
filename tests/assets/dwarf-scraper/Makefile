
CHERISDK?=/cheri/cherisdk

CHERI_CC?=$(CHERISDK)/sdk/bin/clang
CHERI_CFLAGS=--sysroot $(CHERISDK)/rootfs-riscv64-purecap -target riscv64-unknown-freebsd -march=rv64imafdcxcheri -mno-relax

SOURCES := $(wildcard *.c)

TARGETS := $(SOURCES:%.c=riscv_dwarf_%)
all: $(TARGETS)

riscv_dwarf_%: %.c
	$(CHERI_CC) $(CHERI_CFLAGS) -gdwarf-4 -mabi=l64pc128d -o $@ $^

clean:
	rm $(TARGETS)
