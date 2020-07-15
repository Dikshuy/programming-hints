# Hints for C++
## This is a repo for hints of all the problems that I have solved on hacker-rank.

1.
### Factorial of a large number:
Issue of the problem: Large factorials can't be stored even in case of long long int. So, the given below is a idea for solving such cases.

initialize a matrix of a large size ,let's say, 1000;

put its start value as 1 and one other parameter size as 1. Now, as you peform normal multplication update the values.

**Implementation**:
```bash
void extraLongFactorials(int n) {
    int val[1000];
    int size = 1;
    val[0] = 1;
    size = 1;
    for(int i=2; i<=n; i++){
        int carry = 0;
        for(int j=0; j<size; j++){
            int pod = val[j]*i+carry;
            val[j] = pod%10;
            carry = pod/10;
        }
        while(carry){
            val[size] = carry%10;
            carry/=10;
            size++;
        }
    }
    for(int i = size-1; i>= 0; i--)cout << val[i];
}
```
