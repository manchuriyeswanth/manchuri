#include <iostream>
#include <bits/stdc++.h>
using namespace std;

bool subsetsum(int, int, vector<int>&);
int main()
{
    vector<int> arr={3,4,5};
    int target=9;
    cout<<""<<subsetsum(0,target,arr);
}

bool subsetsum(int ind, int target, vector<int>& arr)
{
    if (target ==0)
        return true;
    
    if(target<0 || ind==arr.size())
        return false;
    
    if (subsetsum(ind+1, target-arr[ind], arr))
        return true;

    return subsetsum(ind+1, target, arr);

}