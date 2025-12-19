
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import uuid
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from submission_buffer import MongoSubmissionBuffer

class TestSubmissionBuffer(unittest.TestCase):
    
    @patch('submission_buffer.MongoClient')
    def setUp(self, mock_client):
        self.mock_client = mock_client
        self.mock_db = MagicMock()
        self.mock_collection = MagicMock()
        self.mock_client.return_value.__getitem__.return_value = self.mock_db
        self.mock_db.__getitem__.return_value = self.mock_collection
        
        self.buffer = MongoSubmissionBuffer()

    def test_save_submissions(self):
        submissions = {"file1.py": "print('hello')", "file2.py": "x = 1"}
        
        buffer_id = self.buffer.save_submissions(submissions)
        
        # Verify buffer_id is a UUID
        self.assertTrue(uuid.UUID(buffer_id))
        
        # Verify insert_many was called with correct docs
        call_args = self.mock_collection.insert_many.call_args[0][0]
        self.assertEqual(len(call_args), 2)
        self.assertEqual(call_args[0]['name'], "file1.py")
        self.assertEqual(call_args[1]['name'], "file2.py")
        self.assertEqual(call_args[0]['buffer_id'], buffer_id)

    def test_get_submissions(self):
        buffer_id = "test-uuid"
        self.mock_collection.find.return_value = [
            {"name": "f1", "raw_text": "t1"},
            {"name": "f2", "raw_text": "t2"}
        ]
        
        results = self.buffer.get_submissions(buffer_id)
        
        self.assertEqual(results, {"f1": "t1", "f2": "t2"})
        self.mock_collection.find.assert_called_with({"buffer_id": buffer_id})

    def test_save_preprocessed(self):
        buffer_id = "test-uuid"
        name = "file1.py"
        data = {"tokens": ["a", "b"]}
        
        self.mock_collection.update_one.return_value.modified_count = 1
        
        success = self.buffer.save_preprocessed(buffer_id, name, data)
        
        self.assertTrue(success)
        self.mock_collection.update_one.assert_called_with(
            {"buffer_id": buffer_id, "name": name},
            {"$set": {"preprocessed": data}}
        )

if __name__ == '__main__':
    unittest.main()
