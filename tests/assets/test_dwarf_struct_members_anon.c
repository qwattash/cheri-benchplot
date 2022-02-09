
#include <stdio.h>

typedef struct {
  long z;
} baz_t;

struct foo {
  baz_t a;
  struct {
    int s_a;
    char *s_b;
  };
  union {
    int un_a;
    char *un_b;
  };
};

struct bar {
  struct {
    long a;
    char *b;
  } nested;
};

void show(struct foo *fp)
{
  printf("Foo %d %s\n", fp->s_a, fp->s_b);
}

void show2(struct bar *bp)
{
  printf("Bar %s\n", bp->nested.b);
}

int main(int argc, char *argv[])
{

  struct foo f;
  struct bar b;

  f.s_a = 10;
  f.s_b = "hello";
  show(&f);

  b.nested.b = "hello";
  show2(&b);

  return 0;
}
