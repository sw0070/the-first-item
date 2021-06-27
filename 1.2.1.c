#include<stdio.h>
/*
指针+1=?
指针+n=?
char型 
*/

int main(void){
  char ac[] = {0,1,2,3,4,5,6,7,8,9,};
  char *p = ac;
  char *p1 = &ac[5];
  printf("p = %p\n", p);
  printf("p+1 = %p\n", p+1);  
  printf("p1 - p = %d\n", p1-p);  // 指针的加减
   
  int ai[] = {0,1,2,3,4,5,6,7,8,9,};
  int *q = ai;
  int *q1 = &ai[6];
  printf("q = %p\n", q);
  printf("q+1 = %p\n", q+1);
  printf("q1 - q = %d\n", q1-q);  //指针的加减，得到地址相减/sizeof
  return 0;
}

/*代码块2
有关*p++;
int main(void){
  int ac[] = {0,1,2,3,4,5,6,7,8,9,-1};
  int *p = ac;
  //原有
  //for(i=0; i<sizeof(ac)/sizeof(ac[0]); i++){
  //  printf("%d\n", *(p+i));
  //} 
  while(*p != -1){
    printf("%d\n", *p++);  //取出p所指的哪个数据，完事后顺便把p移到下一个位置去
  }
  return 0;
}

    
    
  












