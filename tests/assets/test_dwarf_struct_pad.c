
#include <stdio.h>

struct bar {
  char *x;
  int y;
  /* expect 4 bytes pad */
};

struct foo {
  char a;
  /* expect 3 bytes pad */
  int b;
  char c;
  /* expect 7 bytes pad */
  struct bar d;
  /* expect 0 bytes pad */
  long e;
};

void show(struct foo *fp)
{
  printf("Foo %d\n", fp->a);
}

int main(int argc, char *argv[])
{

  struct foo f;

  f.a = 10;
  show(&f);

  return 0;
}
