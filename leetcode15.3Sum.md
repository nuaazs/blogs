## 15 3Sum

Given an array `nums` of *n* integers, are there elements *a*, *b*, *c* in `nums` such that *a* + *b* + *c* = 0? Find all unique triplets in the array which gives the sum of zero.

Notice that the solution set must not contain duplicate triplets.

 

**Example 1:**

```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

**Example 2:**

```
Input: nums = []
Output: []
```

**Example 3:**

```
Input: nums = [0]
Output: []
```

 

**Constraints:**

- `0 <= nums.length <= 3000`
- `-105 <= nums[i] <= 105`





## Solutions:

the key idea is the same as the `TwoSum` problem. When we fix the `1st` number, the `2nd` and `3rd` number can be found following the same reasoning as `TwoSum`.



The only difference is that, the `TwoSum` problem of LEETCODE has a **unique solution**. However, in `ThreeSum`, we have multiple duplicate solutions that can be found. Most of the OLE errors happened here because you could've ended up with a solution with so many duplicates.

The naive solution for the duplicates will be using the STL methods like below :

```cpp
std::sort(res.begin(), res.end());
res.erase(unique(res.begin(), res.end()), res.end());
```

But according to my submissions, this way will cause you double your time consuming almostly.

A better approach is that, to jump over the number which has been scanned, no matter it is part of some solution or not.

If the three numbers formed a solution, we can safely ignore all the duplicates of them.

We can do this to all the three numbers such that we can remove the duplicates.

Here's my AC C++ Code:

```cpp
vector<vector<int>> threeSum(vector<int> &num){
    vector<vector<int>> res;
    std::sort(num.begin(), num.end());
    for (int i =0; i<num.size(); i++){
        int target = -num[i];
        int front = i+1;
        int back = num.size() -1;
        while (front < back){
            int sum = num[front] + num[back];
            //Finding answer whick start from number num[i]
            if (sum < target)
                front++;
            else if (sum > target)
                back--;
            else{
                vector<int> triplet = {num[i],num[front],num[back]};
                res.push_back(triplet);

                //Processing duplicates of Number 2
                //Rolling the front pointer to the next different number forwards
                while(front < back && num[front] == triplet[1]) front++;
                
                //Processing duplicates of Number 3
                //Rolling the back pointer to the next different number backwards
                while(front < back && num[back] == triplet[2]) back--;
                
            }
        }
        //Processing duplicates of Number1
        while(i+1 < num.size() && num[i+1] == num[i])
            i++;
    }
    return res;
}
```



```python
def threeSum(self, nums):
    res=[]
    nums.sort()
    for i in xrange(len(nums)-2):
        if i>0 and nums[i] == nums[i-1]:
            continue
        l, r = i+1, len(nums)-1
        while l<r:
            s=nums[i] + nums[l] + nums[r]
            if s<0:
                l += 1
            elif s>0:
                r-=1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l<r and nums[l] == nums[l+1]:
                    l += 1
                while l<r and nums[r] == nums[r-1]:
                    r -=1
                l+=1; r -= 1
    return res
```

