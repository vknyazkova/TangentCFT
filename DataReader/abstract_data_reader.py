from typing import List, Dict


class AbstractDataReader:
    def get_collection(self) -> Dict[str, List[str]]:
        raise NotImplementedError

    def get_query(self):
        raise NotImplementedError
