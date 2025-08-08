
#include <stddef.h>
#include <stdint.h>

/* Lifted from CheriBSD sys/net/mppcc.c struct MPPC_comp_state */
struct foo {
  // 0x0
  uint8_t hist[16384];
  // 0x4000
  uint16_t histptr;
  // 0x4002
  uint16_t hash[8192];
  // 0x8002
};

_Static_assert(sizeof(struct foo) == 0x8002, "Unexpected size");
_Static_assert(offsetof(struct foo, hist) == 0x0, "foo::hist offset");
_Static_assert(offsetof(struct foo, histptr) == 0x4000, "foo::histptr offset");
_Static_assert(offsetof(struct foo, hash) == 0x4002, "foo::hash offset");

struct foo x __attribute__((used));

int main() {
  return 0;
}
