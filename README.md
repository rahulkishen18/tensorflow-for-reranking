## Launching TensorFlow jobs on MCLI

This is a simple training workflow for training a neural reranker. It will:
1. Automatically authenticate with GCP services based on credentials setup with MCLI
2. Pull data from BigQuery
3. Save checkpoints to GCP asynchronously
4. Train a simple neural reranker

### Some Background
I spent most of the time wrangling with GCP services, mainly BigQuery. I'm not super familiar with the ins and outs of BigQuery, so futzing around with how to save the features took a minute, given it's an N x 136 numpy array.

A nice benefit of MCLI is you can "ssh" (not exactly ssh protocol, but effectively exactly the same idea) onto a node and debug directly. I setup a Remote VS Code session on the node, clone down the repo and do some debugging there.

To do that, you just run 
```
mcli interactive {NUMBER OF HOURS FOR INTERACTIVE SESSION} --cluster {CLUSTER YOU WANT TO USE}
```

Then you can play around directly on the box to your hearts content.

### Provisioning environment for GCP authentication
MCLI supports storing secure secrets per user, that are then injected into the users environment. Here I setup [GCP credentials](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/gcp.html) which are injected into the runtime in such a way that standard GCP Python libraries can use them to auto-authenticate. I know it says "GCP Storage", but this is a minsomer, it's for authenticating GCP in general.

### Launching MCLI runs

Once you've debugged your code, you can start launching runs that will spin up resources, execute a training run and tear down resources for you automatically. For this, you can just run 
```
mcli run -f tf-run.yaml
```

That's it!
