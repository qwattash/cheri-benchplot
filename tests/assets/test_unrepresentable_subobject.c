
struct test_simple {
  int skew_offset;
  char large_buffer[((1 << 13) - 7)];
};

struct test_age_softc_layout {
  char before[0x250];
  char cdata[0x6140];
  char after[0x190];
};

struct test_complex {
  int before;
  struct {
    char buf_before[(1 << 13) - 4];
    // 8KiB layout boundary
    char buf_after[(1 << 13) - 5];
  } inner;
  char after[10];
};

struct test_flexible {
  int int_value;
  char flexbuf[];
};

struct test_nested {
  struct test_complex a;
};

int main() {
  struct test_simple a;
  struct test_age_softc_layout c;
  struct test_flexible d;
  struct test_complex e;
  struct test_nested f;

  return 0;
}
