
#include <stdio.h>

struct bar {
  int x;
  int y;
};

struct foo {
  int a;
  char *b;
  struct bar c;
};

void show(struct foo *fp)
{
  printf("Foo %d %s\n", fp->a, fp->b);
}

int main(int argc, char *argv[])
{

  struct foo f;

  f.a = 10;
  f.b = "hello";
  f.c.x = 100;
  f.c.y = 200;
  show(&f);

  return 0;
}
