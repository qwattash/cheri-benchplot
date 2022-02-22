
#include <stdio.h>

struct baz;

typedef struct baz baz_t;

struct baz {
  int x;
  char pad;
  char *y;
};

struct bar {
  baz_t xy;
  char pad;
  char *z;
};

struct foo {
  int a;
  struct bar b;
  struct bar *c;
  struct bar d[0];  /* This should not count towards the nested ptr count */
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
