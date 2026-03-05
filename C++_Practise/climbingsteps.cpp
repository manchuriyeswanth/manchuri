#include <iostream>
#include <bits/stdc++.h>
using namespace std;

int countways(int, vector<int>&);
int main()
{
    int n=4;
    vector<int> dp;
    dp.resize(n+1,-1);
    cout<<"No of Ways:"<<countways(n,dp)<<endl;
    return 0;
}

int countways(int n, vector<int>& dp)
{
    if (n==0 || n==1)
        return 1;
    
    if (dp[n]!=-1)
        return dp[n];
    
    dp[n] = countways(n-1,dp)+ countways(n-2,dp);
    return dp[n];
}