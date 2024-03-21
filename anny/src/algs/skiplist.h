#pragma once
#include <vector>
#include <optional>
#include <random>
#include <cmath>
#include <memory>

namespace anny { namespace experimental {

template <typename Key, typename T, typename Compare = std::less<Key>>
class SkipList
{
public:
	SkipList() = default;
	SkipList(const std::vector<std::pair<Key, T>>& data);

	std::optional<T> find(const Key&);
	void print();

private:
	struct Node;
	using NodePtr = std::unique_ptr<Node>;
	struct Node
	{
		explicit Node(Key key, T value, size_t num_layers)
			: key{key}
			, value{value}
			, next{nullptr}
			, next_layers{num_layers, nullptr}
		{}

		Key key{};
		T value{};
		NodePtr next{ nullptr };  // owning pointer at layer 0
		std::vector<Node*> next_layers;  // non-owning links (shortcuts) to next nodes at upper layers
		size_t num_layers() const { return next.size(); }
	};

	NodePtr m_skiplist{ nullptr };
	std::mt19937 m_gen{ (std::random_device())() };
	size_t m_numLayers{0};
};


template <typename Key, typename T, typename Compare>
SkipList<Key, T, Compare>::SkipList(const std::vector<std::pair<Key, T>>& data)
{
	std::vector<std::pair<Key, T>> sorted_data(data);  // ugly copy
	Compare comp{};
	std::sort(sorted_data.begin(), sorted_data.end(), [comp](auto a, auto b) { return comp(a.first, b.first);  });
	m_numLayers = static_cast<size_t>(1 + log2(sorted_data.size()));
	const size_t N = sorted_data.size();

	std::bernoulli_distribution dis(0.5);  // to flip coin

	// first node (with min key) must contain all layers
	m_skiplist = std::make_unique<Node>(sorted_data[0].first, sorted_data[0].second, m_numLayers);

	std::vector<Node*> prev(m_numLayers, m_skiplist.get());
	for (size_t i = 1; i < sorted_data.size(); i++)
	{
		//detect height of curr node
		size_t curr_layer = 0;
		if (i != sorted_data.size() - 1)
		{
			while ((curr_layer < m_numLayers - 1) && dis(m_gen))
			{
				curr_layer++;
			}
		}
		else
			curr_layer = m_numLayers - 1;

		NodePtr curr_node = std::make_unique<Node>(sorted_data[i].first, sorted_data[i].second, curr_layer + 1);
		Node* curr_ptr = curr_node.get();

		std::cout << "Node with key " << curr_ptr->key << " has " << curr_layer + 1 << " layers.\n";

		for (size_t k = 0; k <= curr_layer; k++)
		{
			prev[k]->next_layers[k] = curr_ptr;
		}
		prev[0]->next = std::move(curr_node);
		for (size_t k = 0; k <= curr_layer; k++)
		{
			prev[k] = curr_ptr;  // update prev
		}

	}

}

template <typename Key, typename T, typename Compare>
std::optional<T> SkipList<Key, T, Compare>::find(const Key& key)
{
	std::cout << "Skiplist: searching for key " << key << std::endl;

	size_t curr_layer = m_numLayers - 1;
	Compare comp{};
	Node* curr_node = m_skiplist.get();
	Node* prev_node{ nullptr };
	while (curr_node && curr_layer >= 0)
	{
		std::cout << "curr_layer = " << curr_layer << ", curr_node key = " << curr_node->key << std::endl;

		if (curr_node->key == key)
			return curr_node->value;
		
		// move along current layer until node key < given key
		if (comp(curr_node->key, key))
		{
			prev_node = curr_node;
			curr_node = curr_node->next_layers[curr_layer];
		}
		else
		{
			// go 1 layer down if possible			
			if (curr_layer == 0)
				break;
			curr_layer--;
			curr_node = prev_node;
		}
	}
	return {};
}

template <typename Key, typename T, typename Compare>
void SkipList<Key, T, Compare>::print()
{
	if (!m_skiplist)
	{
		std::cout << "skiplist is empty.\n";
		return;
	}

	for (size_t curr_layer = 0; curr_layer < m_numLayers; curr_layer++)
	{
		Node* curr_node = m_skiplist.get();
		Key curr_key = curr_node->key;
		Key prev_key = curr_node->key;
		while (curr_node)
		{
			curr_key = curr_node->key;
			size_t dist = (curr_key != prev_key) ? ((curr_key - prev_key - 1) * 2 + 1) : 0;
			for (size_t i = 0; i < dist; i++)
				std::cout << "-";
			std::cout << curr_key;
			curr_node = curr_node->next_layers[curr_layer];
			prev_key = curr_key;
		}
		std::cout << std::endl;
	}
}

}
}