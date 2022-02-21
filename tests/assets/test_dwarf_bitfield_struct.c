
#include <stdio.h>

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


void show(struct foo *fp, struct bar *bp)
{
  printf("Foo %ld %lx\n", fp->x, bp->x);
}

int main(int argc, char *argv[])
{
  struct foo f;
  struct bar b;

  f.x = 10;
  b.x = 20;
  show(&f, &b);

  return 0;
}
