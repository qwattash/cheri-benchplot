
struct bar {
  int x;
  int y;
};

typedef struct baz {
  long z;
} baz_t;

struct forward;

struct foo {
  int a;
  char *b;
  struct bar c;
  const char *d;
  char * const e;
  const volatile void *f;
  int **g;
  int *h[10];
  void (*i)(int, int);
  int (*j)[10];
  baz_t k;
  struct forward *l;
  /* char  (*(*(* m [3])(int))[4])(double); */
};

int main(int argc, char *argv[])
{
  struct foo f;

  return 0;
}
