import unittest

from mgo_v2.cache import GlobalCacheState


class TestCache(unittest.TestCase):
    def test_single_copy_and_slot_consistency(self):
        cache = GlobalCacheState([2, 2])
        cache.place(0, (0, 1), 0, 1)
        self.assertEqual(cache.owner_of((0, 1)), 0)
        cache.assert_consistent()
        rank, slot = cache.evict((0, 1))
        self.assertEqual((rank, slot), (0, 0))
        cache.place(1, (0, 1), 1, 2)
        self.assertEqual(cache.owner_of((0, 1)), 1)
        cache.assert_consistent()


if __name__ == "__main__":
    unittest.main()
