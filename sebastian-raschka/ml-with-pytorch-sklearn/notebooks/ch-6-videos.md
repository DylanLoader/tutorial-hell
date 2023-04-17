# Chapter 6 Decision Trees

## 6.1 Introduction to Decision Trees

Note: All decision trees may be deconstructed into a set of rules, but not all systems of rules may be used to construct decision trees.This is the reason trees are considered interpretable or explainable ML. 


## 6.2 Recursive Algorithms and Big-O

Recursive algorithms are divide and conquer approaches that solves a problem by solving subproblems. 

def some_fun(x:list):
    """Computes the length of an array using recursion

    Args:
        x (list): _description_

    Returns:
        _type_: _description_
    """
    if x == []:
        return 0
    else: 
        return 1 + some_fun(x[1:])

Recursion is limited by the allocation of memory to the stack. If the recursion is too deep, we will hit a stackoverflow error

### Time complexity for Growing the Decision Tree

It can be shown that the optimal split is on the boundary between adjacent examples (similar feature value) with different class labels.

For growing the decision tree we need to sort the values before finding the decision boundaries. To do this need nlogn time. Then with m features we have $O(m \cdot nlogn)$.Assuming we have a perfectly binary tree we have depth $2log_2(n) = n$ splits. We also assume we grow the tree until the nodes are pure and furthermore to the point that each terminal node is a leaf node. For complexity analysis we need to split on the nodes prior to the leaf nodes where each split has time complexity O(m nlogn). We have $O(m n^2 logn)$, asssuming we have the resort at each node. Smart implementations use caching to remove the need to resort at each node and thus have time complexity $O(m nlogn)$ 

Querying the tree:

When performing prediction we only have time complexity of O(log_2(n)) since at each node we are removing half of the remaining nodes to query. 

## 6.3 Types of Decision Trees

#### Decision Tree Psuedocode

GenerateTree(D):

- if $y = 1\ \forall  	\langle x, y \rangle \in \mathbb{D}\ or\ y=0\ \forall \langle x, y \rangle \in \mathbb{D}: Return Tree$
- Else
  - Pick best feature $x_j:$
    - $\mathbb{D_0}\ at\ Child_0:\ x_j = 0 \forall \langle x, y \rangle \in \mathbb{D}$
    - $\mathbb{D_1}\ at\ Child_1:\ x_j = 1 \forall \langle x, y \rangle \in \mathbb{D}$
  
    Return $Node(x_j, GenerateTree(\mathbb{D_0}), GenerateTree(\mathbb{D_1}))$



