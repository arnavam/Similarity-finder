- read the whole code to find the redundant parts

## features:
- add urls history only if there is update in existing ones
- proper display error ( mainly if urls cant be processed )
- vector database & llm ( local and api )
- urls history frontend
- image , video
- find edge cases


## redundant:
- empty submission dict on api call
## risks:
- always usage default encoding when extracting
- names such as macox is used to find top folder
- duplicate function usage ?
- user can upload same file multiple times
- user has no way to remove history urls
- user can create multiple buffers without a limit 
# optional features
- directly send the code to compare / embedding etc.. to server rather than buffer id
- @lru_cache for preprocessing
