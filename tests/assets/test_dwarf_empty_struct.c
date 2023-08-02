
#include <stdio.h>

struct bar {
};

struct foo {
  int a;
  struct bar b;
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
