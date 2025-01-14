
#include <stdio.h>

typedef struct {
  int x;
  int y;
} bar;

typedef struct {
  int a;
  char *b;
  bar c;
} foo;

int main(int argc, char *argv[])
{

  foo f;

  return 0;
}
