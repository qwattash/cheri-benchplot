
struct foo {
  char before;
  int bitfield_a:8;
  int bitfield_b:24;
  char after;
  long int x;
};

struct bar {
  int before;
  int bitfield_a:3;
  int bitfield_b:4;
  long int x;
};

struct etherip_header {
  unsigned int eip_resvl:4, eip_ver:4;
  unsigned char eip_resvh;
} __attribute__((packed));


int main(int argc, char *argv[])
{
  struct etherip_header h;
  struct foo f;
  struct bar b;

  return 0;
}
