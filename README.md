## Hints for hacker-rank problems

### Larry's array:
Issue of the problem: Test whether array can be sorted by swaping three items at a time in order: ABC -> BCA -> CAB -> ABC. [Link](https://www.hackerrank.com/challenges/larrys-array/problem?isFullScreen=true)

Hint: Check out the number of inversions. The given below implementation seems to be simple but there is an awesome logic behind this. Refer to these paper's for understanding the logic behind it. [Paper1](https://www.cs.bham.ac.uk/~mdr/teaching/modules04/java2/TilesSolvability.html) and [Paper2](http://kevingong.com/Math/SixteenPuzzle.html)

**Implementation**:
```bash
string larrysArray(vector<int> A) {
    int n=A.size();
    int sum=0;
    for(int i = 0; i < n; i++){
        for(int j = i+1; j < n; j++){
            if(A[j] < A[i]){
                sum += 1;
            } 
        }
    }
    if(sum%2==0){
        return "YES";
    }
    else{
        return "NO";
    }
} 
```
### Minimum Loss:
Issue of the problem: Time complexity issue in case of larger values. [link](https://www.hackerrank.com/challenges/minimum-loss/problem)

Hint: Sort the array, and then check the difference of adjacent pairs, if its less than ur last min value, update it only if the index of those pairs are in same way in original array.

**Implementation**:
```bash
    int n;
    cin >> n;
    vector<double> sorted(n);
    map<double, int> arr;
    for (int i = 0; i < n; ++i) {
        double x;
        cin >> x;
        sorted[i] = x;
        arr[x] = i;
    }
    sort(sorted.begin(), sorted.end());
    double min = INT_MAX;
    for (int i = 0; i < n - 1; ++i) {
        double x = sorted[i + 1] - sorted[i];
        if (x < min) {
            int first = arr[sorted[i]];
            int second = arr[sorted[i + 1]];
            if (second < first) {
                min = x;
            }
        }
    }
    cout << long(min);
```
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

### 3D surface area:
Issue of the problem: [link](https://www.hackerrank.com/challenges/3d-surface-area/problem?isFullScreen=true)

Hint: The base of the Figure will always contribute to the total surface area of the figure. Now, to calculate the area contributed by the walls, we will take out the absolute difference between the height of two adjacent wall. The difference will be the contribution in the total surface area.

**Implementation**
```bash
int contribution_height(int current, int previous) { 
    return abs(current - previous); 
}
int surfaceArea(vector<vector<int>> A) {
    int ans = 0; 
    int N = A.size();
    int M = A[0].size();
     
    for (int i = 0; i < N; i++) { 
        for (int j = 0; j < M; j++) {
            int up = 0; 
            int left = 0; 
  
            if (i > 0) 
                up = A[i - 1][j]; 
  
            if (j > 0) 
                left = A[i][j - 1]; 
   
            ans += contribution_height(A[i][j], up)  
                    + contribution_height(A[i][j], left); 
 
            if (i == N - 1) 
                ans += A[i][j]; 
            if (j == M - 1) 
                ans += A[i][j]; 
        } 
    } 
  
    // Adding the contribution by the base and top of the figure 
    ans += N * M * 2; 
    return ans;
}
```

### Absolute Permutation
Issue: Represents the smallest lexicographically smallest permutation of natural numbers, such that |pos[i]-i|=k.  [link](https://www.hackerrank.com/challenges/absolute-permutation/problem?isFullScreen=true)

Hint: Distribute into k and swap between 2k.

**Implementation**:
```bash
vector<int> absolutePermutation(int n, int k) {
    vector<int> pos(n);
    for(int i=0; i<n; i++){
        pos[i]=i+1;
    }
    vector<int> permutation(n);
    if(k!=0){
        if(n%k!=0 || (n/k)%2!=0 || k>n/2){
        return {-1};
        }
        for(int m=0; m<n; m=m+2*k){
            for(int j=0;j<k;j++){
                swap(pos[m+j], pos[m+j+k]);
            }
        }
    }
    permutation = pos;

    return permutation;
}
```

### Ordering the team
Issue: Check whether the teams can be ranked on the basis of three parameters. [link](https://www.hackerrank.com/contests/pt-test-3/challenges/ordering-the-team/problem)

Hint: Make a 2d vector and sort it. Then do comparisons in rows and change the bool if not possible.

**Implementation**:
```bash
int n;
    cin>>n;
    vector<vector<int>> v(n);
    for (int i=0;i<n;i++)
    {
        int a, b, c;
        cin>>a>>b>>c;
        v[i].push_back(a);
        v[i].push_back(b);
        v[i].push_back(c);
    }
    sort(v.begin(), v.end());
  
    bool ans = true;
    for (int i=0;i<n-1;i++){    
        int count = 0;
        for (int j=0;j<3;j++){
            if (v[i][j] < v[i+1][j])
                count++;
            else if (v[i][j] > v[i+1][j])
                ans = false;
        }
        if (count == 0)
            ans = false;
    }
    if (ans)
        cout<<"Yes";
    else
        cout<<"No";
```bash
