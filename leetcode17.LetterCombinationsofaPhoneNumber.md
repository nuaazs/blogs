## 17. Letter Combinations of a Phone Number

Given a string containing digits from `2-9` inclusive, return all possible letter combinations that the number could represent. Return the answer in **any order**.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

![](https://shengbucket.oss-cn-hangzhou.aliyuncs.com/pics/PsKev.jpg)

 

**Example 1:**

```
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

**Example 2:**

```
Input: digits = ""
Output: []
```

**Example 3:**

```
Input: digits = "2"
Output: ["a","b","c"]
```

 

**Constraints:**

- `0 <= digits.length <= 4`
- `digits[i]` is a digit in the range `['2', '9']`.





### Iterative C++ solution in 0ms

```cpp
class Solution{
    public:
    const vector<string> pad = {
                "", "", "abc", "def", "ghi", "jkl",
        "mno", "pqrs", "tuv", "wxyz"
    };
    
    vector<string> letterCombinations(string digits){
        if (digits.empty()) return {};
        vector<string> result;
        result.push_back("");
        
        for(auto digit: digits){
            vector<string> tmp;
            for (auto candidate:pad[digit - '0']){
                for(auto s:result){
                    tmp.push_back(s+candidate);
                }
            }
            result.swap(tmp);
        }
        return result;
    }
};
```

More clear:

```cpp
class Solution {
public:
vector<string> letterCombinations(string digits){
    if (digits.empty()) return {};
    vector<string> res;
    res.push_back("");
    map<char,string> map;
    map = {{'2',"abc"},{'3',"def"},{'4',"ghi"},{'5',"jkl"},{'6',"mno"},{'7',"pqrs"},{'8',"tuv"},{'9',"wxyz"}};
    for (int i=0; i<digits.size(); ++i){
        if(digits[i]<'2' || digits[i]>'9'){
            continue;
        }
        
        string cand = map[digits[i]];
        vector<string> tmp;
        for (int j=0;j < cand.size(); ++j){
            for (int k=0; k<res.size(); ++k){
                tmp.push_back(res[k] + cand[j]);
            }
        }
        res = tmp;
    }
    return res;
}
};
```



```python
def letterCombinations(self, digits):
    dict = {'2':"abc", '3':"def", '4':"ghi", '5':"jkl", '6':"mno", '7': "pqrs", 
        '8':"tuv", '9':"wxyz"}
    cmb = [''] if digits else []
    for d in digits:
        cmb = [p+q for p in cmb for q in dict[d]]
    return cmb
```

