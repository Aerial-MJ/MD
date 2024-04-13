# 读入字符串

```c++
#include <stdio.h>

void main(void){
	char c1[]={'I',' ','a','m',' ','h','a','p','p','y'};
	char c2[]="I am happy";
	int i1=sizeof(c1);
	int i2=sizeof(c2);
	printf("%d\n",i1);
	printf("%d\n",i2);
}

```

**结果：10 11**