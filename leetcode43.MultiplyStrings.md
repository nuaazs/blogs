## 43 Multiply Strings

Medium

Given two non-negative integers `num1` and `num2` represented as strings, return the product of `num1` and `num2`, also represented as a string.

**Note:** You must not use any built-in BigInteger library or convert the inputs to integer directly.

 

**Example 1:**

```
Input: num1 = "2", num2 = "3"
Output: "6"
```

**Example 2:**

```
Input: num1 = "123", num2 = "456"
Output: "56088"
```

 

**Constraints:**

- `1 <= num1.length, num2.length <= 200`
- `num1` and `num2` consist of digits only.
- Both `num1` and `num2` do not contain any leading zero, except the number `0` itself.

### Solution

```cpp
class Solution {
public:
    string multiply(string n1, string n2) {
        int i = 0, maxSize = n1.size() + n2.size(), prod = 0, j;
        string res(maxSize, '0');
        while (i++ < n1.size() || prod) {
            j = 0;
            while (j++ < n2.size() || prod) {
                prod = (i <= n1.size() ? (n1[n1.size() - i] - 48) : 0) 
                    * (j <= n2.size() ? (n2[n2.size() - j] - 48) : 0) 
                    + prod 
                    + res[maxSize - j - i + 1] 
                    - 48;
                res[maxSize - j + 1 - i] = (prod % 10) + 48;
                prod /= 10;
            }
        }
        i = -1;
        while (++i < maxSize - 1 && res[i] == '0') {}
        return res.substr(i);
    }
};
```



```cpp
class Solution {
public:
    string res;
    string multiply(string num1, string num2) {
        //handle edge-case where the produ t is 0
        if (num1 == "0" || num2 =="0") return "0";

        //num1.size() + num2.size() == max no. of digits
        vector<int> num(num1.size() + num2.size(),0);
        
        //build the number by multiplying one digit at the time
        for (int i=num1.size() -1 ; i>=0; --i){
            for (int j=num2.size() -1;j>=0;--j){
                num[i+j+1] += (num1[i] - '0') * (num2[j]-'0');
                num[i+j] += num[i+j+1] / 10;
                num[i+j+1] %= 10;
            }
        }
        //skip leading 0's
        int i=0;
        while(i<num.size() && num[i]==0) ++i;
        
        //transform the vector to a string
        while(i<num.size()) res.push_back(num[i++] + '0');
        return res;
    }
};
```



