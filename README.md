## Hints for hacker-rank problems

### Factorial of a large number:
Issue of the problem: Large factorials can't be stored even in case of long long int. So, the given below is a idea for solving such cases.

Hint: Iinitialize a matrix of a large size ,let's say, 1000. Put its start value as 1 and one other parameter size as 1. Now, as you peform normal multplication update the values.

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

### Queen's attack:
Problem: Given the queen's position and the locations of all the obstacles, find and print the number of squares the queen can attack from her position at (r_q, c_q).

Hint: Initialize the distances from the current position to the end of the chessboard in every direction to its actual distance. Then check along every direction and when any obstacle comes in front, set that distance as the value along that direction.

**Implemenatation**:
```bash
int queensAttack(int n, int k, int r_q, int c_q, vector<vector<int>> obstacles) {

queen_row = r_q
queen_column = c_q

top = n - queen_row
bottom = queen_row - 1
right = n - queen_column
left = queen_column - 1

top_left = min(n - queen_row, queen_column - 1)
top_right = n - max(queen_column, queen_row)
bottom_left = min(queen_row, queen_column) - 1
bottom_right = min(queen_row - 1, n - queen_column)

for a0 in xrange(k):
    obstacle_row = obstacles[a0][0]
    obstacle_column = obstacles[a0][1]

    if obstacle_row == queen_row:
        if obstacle_column > queen_column:
            top = min(top, obstacle_column - queen_column - 1)
        else:
            bottom = min(bottom, queen_column - obstacle_column - 1)
 
    elif obstacle_column == queen_column:
        if obstacle_row > queen_row:
            right = min(right, obstacle_row - queen_row - 1)
        else:
            left = min(left, queen_row - obstacle_row - 1)

    elif abs(obstacle_column - queen_column) == abs(obstacle_row - queen_row):
   
        if obstacle_column > queen_column and obstacle_row > queen_row:
            top_right = min(top_right, obstacle_column - queen_column - 1)
      
        elif obstacle_column > queen_column and obstacle_row < queen_row:
            bottom_right = min(bottom_right, obstacle_column - queen_column - 1)
        
        elif obstacle_column < queen_column and obstacle_row > queen_row:
            top_left = min(top_left, queen_column - obstacle_column - 1)
       
        elif obstacle_column < queen_column and obstacle_row < queen_row:
            bottom_left = min(bottom_left, queen_column - obstacle_column - 1)
    
            
print top + bottom + right + left + top_left + top_right + bottom_left + bottom_right 
}
```

### Oraginizing Containers of balls:
Issue of the problem: [link](https://www.hackerrank.com/challenges/organizing-containers-of-balls/problem?isFullScreen=false)

Hint: 1. Make a vector of capacity of every box  2. Make a vector of all the balls  3. Sort both of them 

Compare both the vectors. If same then possible else impossible.

**Implementation**:
```bash
string organizingContainers(vector<vector<int>> container){
    vector<int> capacity;
    vector<int> balls;
    for(unsigned int i=0; i<container.size(); i++){
        int cols = 0;
        int rows = 0;
        for(unsigned int j=0; j<container.size(); j++){
            cols += container[i][j];
            rows += container[j][i];
        }
        balls.push_back(cols);
        capacity.push_back(rows);
    }   
    sort(balls.begin(), balls.end());
    sort(capacity.begin(), capacity.end());
    
    if(balls == capacity){
        return "Possible";
    }
    else{
        return "Impossible";
    }

}
```
