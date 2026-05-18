# GitHub CLI Commands Reference

Quick reference for `gh` commands used in support workflow.

## Fetching Issues

```bash
# Get issue with all details
gh issue view NUMBER --repo talmolab/sleap --json number,title,body,author,createdAt,comments,labels,state

# Get issue body only (for quick reading)
gh issue view NUMBER --repo talmolab/sleap --json body -q '.body'

# Get issue comments
gh issue view NUMBER --repo talmolab/sleap --json comments -q '.comments[].body'

# List recent issues
gh issue list --repo talmolab/sleap --limit 10 --json number,title,author,createdAt
```

## Fetching Discussions

```bash
# Get discussion details
gh api repos/talmolab/sleap/discussions/NUMBER

# Get discussion body
gh api repos/talmolab/sleap/discussions/NUMBER -q '.body'

# Get discussion comments
gh api repos/talmolab/sleap/discussions/NUMBER -q '.comments.nodes[].body'

# List recent discussions
gh api repos/talmolab/sleap/discussions --jq '.[] | {number, title, author: .user.login}'
```

## Posting Replies

**Issues:**
```bash
gh issue comment NUMBER --repo talmolab/sleap --body "Your response here"

# Or from file
gh issue comment NUMBER --repo talmolab/sleap --body-file RESPONSE_DRAFT.md
```

**Discussions:**
```bash
# Need to use API for discussions
gh api repos/talmolab/sleap/discussions/NUMBER/comments -X POST -f body="Your response"
```

## Searching

```bash
# Search issues by keyword
gh issue list --repo talmolab/sleap --search "keyword" --json number,title

# Search with labels
gh issue list --repo talmolab/sleap --label "bug" --json number,title

# Search discussions
gh api repos/talmolab/sleap/discussions -q '.[] | select(.title | test("keyword"; "i")) | {number, title}'
```

## Checking Both Issues AND Discussions

Users often post in the wrong category. Always check both:

```bash
# Try issue first, fall back to discussion
gh issue view NUMBER --repo talmolab/sleap 2>/dev/null || \
gh api repos/talmolab/sleap/discussions/NUMBER 2>/dev/null || \
echo "Not found as issue or discussion"
```

## Downloading Images

Images in posts use GitHub's CDN format:
```
https://user-images.githubusercontent.com/...
https://github.com/user-attachments/assets/...
```

Download all images from a post:
```bash
# Extract and download image URLs
grep -oE 'https://[^)]+\.(png|jpg|jpeg|gif)' post.md | while read url; do
  wget -q -P images/ "$url"
done
```

## Labels

Common labels in talmolab/sleap:
- `bug` - Bug reports
- `enhancement` - Feature requests
- `question` - Usage questions
- `documentation` - Doc improvements needed
- `good first issue` - Beginner-friendly

Add a label:
```bash
gh issue edit NUMBER --repo talmolab/sleap --add-label "label-name"
```

## Closing Issues

```bash
# Close with comment
gh issue close NUMBER --repo talmolab/sleap --comment "Closing as resolved. Feel free to reopen if needed!"

# Close as not planned
gh issue close NUMBER --repo talmolab/sleap --reason "not planned" --comment "Thanks for the suggestion!"
```
