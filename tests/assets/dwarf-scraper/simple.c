
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

struct foo f __attribute__((used));

int main()
{
  return 0;
}
