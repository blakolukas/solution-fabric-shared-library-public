# """Check which agent in a list has a given workflow available."""

# import requests
# from core.task import task


# @task(
#     outputs=["agent_url"],
#     output_types={"agent_url": "str"},
#     display_name="Get Workflow Agent",
#     description="Find which agent in a list has a specific workflow available",
#     category="workflow",
#     parameters={
#         "workflow_name": {
#             "type": "str",
#             "required": True,
#             "description": "Name of the workflow to look for",
#         },
#         "agent_urls": {
#             "type": "list",
#             "required": True,
#             "description": "List of FabricFlow agent base URLs to search",
#         },
#     },
# )
# def get_workflow_agent(workflow_name: str, agent_urls: list) -> str:
#     """
#     Iterate over agent URLs and return the first one that has the workflow.
#
#     Returns:
#         agent_url: URL of the agent that has the workflow, or empty string if not found
#     """
#     for url in agent_urls:
#         try:
#             response = requests.get(f"{url}/workflows", timeout=5)
#             if response.status_code == 200:
#                 data = response.json()
#                 names = [w.get("name") for w in data.get("workflows", [])]
#                 if workflow_name in names:
#                     return url
#         except requests.RequestException:
#             continue
#     return ""