sudo kubectl autoscale deployment frontend -n hipster --min=1 --max=8 --cpu-percent=80
sudo kubectl autoscale deployment cartservice -n hipster --min=1 --max=8 --cpu-percent=80
sudo kubectl autoscale deployment adservice -n hipster --min=1 --max=8 --cpu-percent=80
sudo kubectl autoscale deployment checkoutservice -n hipster --min=1 --max=8 --cpu-percent=80
sudo kubectl autoscale deployment emailservice -n hipster --min=1 --max=8 --cpu-percent=80
sudo kubectl autoscale deployment productcatalogservice -n hipster --min=1 --max=8 --cpu-percent=80
sudo kubectl autoscale deployment recommendationservice -n hipster --min=1 --max=8 --cpu-percent=80
sudo kubectl autoscale deployment paymentservice -n hipster --min=1 --max=8 --cpu-percent=80
sudo kubectl autoscale deployment shippingservice -n hipster --min=1 --max=8 --cpu-percent=80
sudo kubectl autoscale deployment currencyservice -n hipster --min=1 --max=8 --cpu-percent=80

sudo kubectl delete hpa --all -n hipster