#include <iostream>
#include <bits/stdc++.h>
using namespace std;

void bt(int, vector<int>&, vector<int>&, vector<vector<int>>&);
void bt_repeat(int, vector<int>&, vector<int>&, vector<vector<int>>&);

int main()
{
    vector<int> nums={1,2,2,3};
    vector<int> current;
    vector<vector<int>> result;

    //bt(0, nums, current, result);
        bt_repeat(0, nums, current, result);
    for (int i=0;i<result.size();i++)
    {
        cout<<"[";
        for (int j=0;j<result[i].size();j++)
        {
            cout<<""<<result[i][j]<<",";
        }
        cout<<"]"<<endl;
    }

}

void bt(int ind, vector<int>& nums, vector<int>& current, vector<vector<int>>& result)
{   
    if (ind == nums.size())
    {
        result.push_back(current);
        return;
    }
    current.push_back(nums[ind]);
    bt(ind+1, nums, current, result);
    current.pop_back();
    bt(ind+1, nums, current, result);
}

void bt_repeat(int index,
               vector<int>& nums,
               vector<int>& current,
               vector<vector<int>>& result)
{
    if (index == nums.size()) {
        result.push_back(current);
        return;
    }

    // INCLUDE current element
    current.push_back(nums[index]);
    bt_repeat(index + 1, nums, current, result);
    current.pop_back();

    // EXCLUDE current element AND skip duplicates
    int nextIndex = index + 1;
    while (nextIndex < nums.size() &&
           nums[nextIndex] == nums[index]) {
        nextIndex++;
    }

    bt_repeat(nextIndex, nums, current, result);
}
