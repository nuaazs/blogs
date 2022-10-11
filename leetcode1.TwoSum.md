## 1. Two Sum

Given an array of integers `nums` and an integer `target`, return *indices of the two numbers such that they add up to `target`*.

You may assume that each input would have ***exactly\* one solution**, and you may not use the *same* element twice.

You can return the answer in any order.

 

**Example 1:**

```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Output: Because nums[0] + nums[1] == 9, we return [0, 1].
```

**Example 2:**

```
Input: nums = [3,2,4], target = 6
Output: [1,2]
```

**Example 3:**

```
Input: nums = [3,3], target = 6
Output: [0,1]
```

 

**Constraints:**

- `2 <= nums.length <= 103`
- `-109 <= nums[i] <= 109`
- `-109 <= target <= 109`
- **Only one valid answer exists.**



### Solution

#### Approach 1: Brute Force

The brute force approach is simple. Loop through each element x and find if there is another value that equals to `target - x`

```java
public int[] twoSum(int[] nums, int target){
    for(int i=0; i<nums.length;i++){
        for (int j=i+1; j<nums.length; j++){
            if (nums[j]= target - nums[i]){
                return new int[]{i,j};
            }
        }
    }
    throw new IllegalArgumentException("NO two sum solution");
}
```



**Complexity Analysis**

- Time complexity : O(n^2). For each element, we try to find its complement by looping through the rest of array which takes O(n) time. Therefore, the time complexity is O(n^2)

- Space complexity : O(1)

  

------

#### Approach 2: Two-pass Hash Table

To improve our run time complexity, we need a more efficient way to check if the complement exists in the array. If the complement exists, we need to look up its index. **What is the best way to maintain a mapping of each element in the array to its index? A hash table.**

We reduce the look up time from O(n) to O(1) by **trading space for speed. A hash table is built exactly for this purpose**, it supports fast look up in *near* constant time. I say "near" because if a collision occurred, a look up could degenerate to O(n) time. But look up in hash table should be amortized O(1) time as long as the hash function was chosen carefully.

A simple implementation uses two iterations. In the first iteration, we add each element's value and its index to the table. Then, in the second iteration we check if each element's complement (target - nums[i] )exists in the table. Beware that the complement must not be nums[i] itself!

```java
public int[] towSum(int[] nums, int target){
    Map<Integer, Integer> map = new HashMap<>();
    for (int i=0; i<nums.length; i++){
        map.put(nums[i],i);
    }
    for (int i =0;i<nums.length; i++){
        int complement = target - nums[i];
        for(map.containsKey(complement) && map.get(complement)!=i){
            return new int[]{i, map.get(complement)};
        }
    }
    throw new IllegalArgumentException("NO two sum solution");
}
```





**Complexity Analysis:**

- Time complexity : O(n) We traverse the list containing n elements exactly twice. Since the hash table reduces the look up time to O(1) , the time complexity is O(n) .

- Space complexity : O(n) . The extra space required depends on the number of items stored in the hash table, which stores exactly n  elements.

  

------

#### Approach 3: One-pass Hash Table

It turns out we can do it in one-pass. **While we iterate and inserting elements into the table, we also look back to check if current element's complement already exists in the table.** If it exists, we have found a solution and return immediately.

```java
public int[] twoSum(int[] nums, int target){
    Map<Integer, Integer> map = new HashMap<>{};
    for(int i=0; i<nums.length; i++){
        int complement = target - nums[i];
        if(map.containsKey(complment)){
            return new int[] {map.get(complement),i};
        }
        map.put(nums[i],i);
    }
    throw new IllegalArgumentException("No two sum solution");
}
```

```cpp
vector<int> twoSum(vector<int> &numbers, int target){
    //Key is the number and value is its index in the vector.
    unordered_map<int, int> hash;
    vector<int> result;
    for(int i =0; i<numbers.size(); i++){
        int numberToFind = target - numbers[i];
        
        // if numberToFind is found in map, return them
        if(hash.find(numberToFind)!=hash.end()){
            
    
            result.push_back(hash[numberToFind]);
            result.push_back(i);
            return result;
        }
        //number was not found. Put it in the map.
        hash[numbers[i]] = i;
    }
    return result;
}
```





**Complexity Analysis:**

- Time complexity : O(n)*O*(*n*). We traverse the list containing n*n* elements only once. Each look up in the table costs only O(1)*O*(1) time.
- Space complexity : O(n)*O*(*n*). The extra space required depends on the number of items stored in the hash table, which stores at most n*n* elements.

```python
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        h = {}
        for i,num in enumerate(nums):
            n = target - num
            if n not in h:
                h[num = i]
            else:
                return [h[n],i]
```

