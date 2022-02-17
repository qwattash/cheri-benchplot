
#include <stdio.h>

typedef int foo_int;
typedef foo_int bar_int;

typedef struct foo {
  bar_int x;
} foo_t;
typedef foo_t bar_t;

void show(bar_t *fp)
{
  printf("Foo %d %s\n", fp->x);
}

int main(int argc, char *argv[])
{
  bar_t f;

  f.x = 10;
  show(&f);

  return 0;
}
