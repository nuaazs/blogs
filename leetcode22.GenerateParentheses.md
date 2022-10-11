## 22. Generate Parentheses

Given `n` pairs of parentheses, write a function to *generate all combinations of well-formed parentheses*.

**Example 1:**

```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

**Example 2:**

```
Input: n = 1
Output: ["()"]
```

 

**Constraints:**

- `1 <= n <= 8`







### Solutions

#### Approach 1: Brute Force

**Intuition**

We can generate all 2^{2n}22*n* sequences of `'('` and `')'` characters. Then, we will check if each one is valid.

**Algorithm**

To generate all sequences, we use a recursion. All sequences of length `n` is just `'('` plus all sequences of length `n-1`, and then `')'` plus all sequences of length `n-1`.

To check whether a sequence is valid, we keep track of `balance`, the net number of opening brackets minus closing brackets. If it falls below zero at any time, or doesn't end in zero, the sequence is invalid - otherwise it is valid.

```python
class Solution(object):
    def generateParenthesis(self, n):
        def generate(A = []):
            if len(A) == 2*n:
                if valid(A):
                    ans.append("".join(A))
                else:
                    A.append('(')
                    generate(A)
                    A.pop()
                    A.append(')')
                    generate(A)
                    A.pop()
        def valid(A):
            bal = 0
            for c in A:
                if c =='(':bal += 1
                else:bal-=1
                if bal<0:return False
            return bal == 0
        ans = []
        generate()
        return ans
```

```java
class Solution{
    public List<String> generateParenthesis(int n){
        List<String> combinations = new ArrayList();
        generateAll(new char[2*n], 0, combinations);
        return combinations;
    }
    public void generateAll(char[] current, int pos, List<String> result){
        if(pos == current.length){
            if (valid(current))
                result.add(new String(current));
        } else {
            current[pos]='(';
            genereateAll(current, pos+1, result);
            current[pos]=')';
            generateAll(current, pos+1, result);
        }
    }
    public boolean valid(char[] current){
        int balance = 0;
        for (char c:current){
            if (c == '(') balance ++;
            else balance--;
            if (balance<0) return false;
        }
        return (balance == 0);
    }
}
```





**Complexity Analysis**

- Time Complexity : O(2^{2n}n)*O*(22*n**n*). For each of 2^{2n}22*n* sequences, we need to create and validate the sequence, which takes O(n)*O*(*n*) work.
- Space Complexity : O(2^{2n}n)*O*(22*n**n*). Naively, every sequence could be valid. 



#### Approach 2: Backtracking

**Intuition and Algorithm**

Instead of adding `'('` or `')'` every time as in [Approach 1](https://leetcode.com/problems/generate-parentheses/solution/#approach-1-brute-force), let's only add them when we know it will remain a valid sequence. We can do this by keeping track of the number of opening and closing brackets we have placed so far.

We can start an opening bracket if we still have one (of `n`) left to place. And we can start a closing bracket if it would not exceed the number of opening brackets.

```python
class Solution(object):
    def generateParenthesis(self, N):
        ans = []
        def backtrack(S='', left=0, right=0):
            if len(S) == 2*N:
                ans.append(S)
                return
            if left<N:
                backtrack(S+'(',left+1,right)
            if right<left:
                backtrack(S+')', left, right+1)
        backtrack()
        return ans
```



```java
class Solution{
    public List<String> generateParenthesis(int n){
        List<String> ans = new ArrayList();
        backtrack(ans,"",0,0,n);
        return ans;
    }
    public void backtrack(List<String> ans, String cur, int open, int close, int max){
        if(cur.length() == max *2){
            ans.add(cur);
            return;
        }
        
        if(open<max)
            backtrack(ans, cur+"(", open+1, close, max);
        if(close<open)
            backtrack(ans, cur+")", open, close+1, max);
    }
}
```

**Complexity Analysis**

Our complexity analysis rests on understanding how many elements there are in `generateParenthesis(n)`. This analysis is outside the scope of this article, but it turns out this is the `n`-th Catalan number ![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/C52yF.jpg), which is bounded asymptotically by ![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/y1xWW.jpg)

- Time Complexity : ![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/P15gy.jpg). Each valid sequence has at most `n` steps during the backtracking procedure.
- Space Complexity :![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/0iPnz.jpg), as described above, and using O(n)*O*(*n*) space to store the sequence.



#### Approach 3: Closure Number

**Intuition**

To enumerate something, generally we would like to express it as a sum of disjoint subsets that are easier to count.

Consider the *closure number* of a valid parentheses sequence `S`: the least `index >= 0` so that `S[0], S[1], ..., S[2*index+1]` is valid. Clearly, every parentheses sequence has a unique *closure number*. We can try to enumerate them individually.

**Algorithm**

For each closure number `c`, we know the starting and ending brackets must be at index `0` and `2*c + 1`. Then, the `2*c` elements between must be a valid sequence, plus the rest of the elements must be a valid sequence.

```python
class Solution(object):
    def genereateParenthesis(self, N):
        if N == 0: return['']
        ans = []
        for c in xrange(N):
            for left in self.generateParenthesis(c):
                for right in self.generateParenthesis(N-1-c):
                    ans.append('({}){}'.format(left,right))
        return ans
```



```java
class Solution{
    public List<String> generateParenthesis(int n){
        List<String> ans = new ArrayList();
        if (n == 0){
            ans.add("");
        } else {
            for (int c = 0; c<n; ++c)
                for (String left: generateParenthesis(c))
                    for (String right: generateParenthesis(n-1-c))
                        ans.add("("+left+")"+right);
        }
        return ans;
    }
}
```



**Complexity Analysis**

- The analysis is similar to [Approach 2](https://leetcode.com/problems/generate-parentheses/solution/#approach-2-backtracking).





The idea is intuitive. Use two integers to count the remaining left parenthesis (n) and the right parenthesis (m) to be added. At each function call add a left parenthesis if n >0 and add a right parenthesis if m>0. Append the result and terminate recursive calls when both m and n are zero.

```cpp
class Solution{
    public:
    vector<string> generateParenthesis(int n){
        vector<string> res;
        addingpar(res,"",n,0);
        return res;
    }
    void addingpar(vector<string> &v, string str, int n, int m){
        if(n == 0 && m == 0){
            v.push_back(str);
            return;
        }
        if(m>0){addingpar(v,str+")",n,m-1);}
        if(n>0){addingpar(v,str+"(",n-1,m+1);}
    }
};
```



