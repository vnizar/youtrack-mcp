from mcp.server.fastmcp import FastMCP
import httpx
import os
import requests
import argparse
from typing import Optional, Dict, Any, List

# Initialize FastMCP server
mcp = FastMCP("youtrack")

class YouTrackAPI:
    def __init__(self, base_url: str, token: str):
        """
        Initialize YouTrack API client
        
        Args:
            base_url (str): Your YouTrack instance URL (e.g., 'https://your-instance.youtrack.cloud')
            token (str): Your permanent token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def get_issue(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific issue
        
        Args:
            issue_id (str): The ID of the issue (e.g., 'PROJ-123')
            
        Returns:
            Optional[Dict[str, Any]]: Issue information or None if not found
        """
        url = f"{self.base_url}/api/issues/{issue_id}?fields=$type,id,summary,description"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching issue {issue_id}: {str(e)}")
            return None

    def post_comment(self, issue_id: str, text: str, permitted_groups: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Post a comment to an issue
        
        Args:
            issue_id (str): The ID of the issue (e.g., 'PROJ-123')
            text (str): The comment text
            permitted_groups (Optional[List[str]]): List of group IDs that can see this comment
            
        Returns:
            Optional[Dict[str, Any]]: Comment information or None if there was an error
        """
        url = f"{self.base_url}/api/issues/{issue_id}/comments"
        
        payload = {
            "text": text
        }
        
        if permitted_groups:
            payload["visibility"] = {
                "permittedGroups": [{"id": group_id} for group_id in permitted_groups],
                "$type": "LimitedVisibility"
            }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error posting comment to issue {issue_id}: {str(e)}")
            return None

def parse_args():
    parser = argparse.ArgumentParser(description="YouTrack MCP Server")
    parser.add_argument(
        "--youtrack-url",
        required=True,
        help="YouTrack instance URL (e.g., 'https://your-instance.youtrack.cloud')"
    )
    parser.add_argument(
        "--youtrack-token",
        required=True,
        help="YouTrack API token"
    )
    return parser.parse_args()

# Global YouTrack API instance
yt = None

@mcp.tool()
async def get_issue(issue_id: str) -> Dict:
    """
    Get information about a specific issue based on issue ID
    
    Args:
        issue_id (str): The ID of the issue (e.g., 'PROJ-123')
        
    Returns:
        Optional[Dict[str, Any]]: Issue information or None if not found
    """
    issue = yt.get_issue(issue_id)
    if not issue or not issue.get('id'):
        return {"error": "Issue not found"}
    
    if not issue.get('description'):
        return {"error": "Description not found"}
    
    return issue

@mcp.tool()
async def post_comment(issue_id: str, text: str, permitted_groups: Optional[List[str]] = None) -> Dict:
    """
    Post a comment to an issue
    
    Args:
        issue_id (str): The ID of the issue (e.g., 'PROJ-123')
        text (str): The comment text
        permitted_groups (Optional[List[str]]): List of group IDs that can see this comment
        
    Returns:
        Dict: Response from the API or error message
    """
    result = yt.post_comment(issue_id, text, permitted_groups)
    if result is None:
        return {"error": "Failed to post comment"}
    return result

if __name__ == "__main__":
    args = parse_args()
    
    # Initialize YouTrack API with command line arguments
    yt = YouTrackAPI(
        base_url=args.youtrack_url,
        token=args.youtrack_token
    )
    
    # Initialize and run the server
    mcp.run(transport='stdio')
