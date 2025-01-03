def get_deployment_from_name_in_hpa(deployment_name_in_hpa: str) -> str:
    return deployment_name_in_hpa[:-4]  # remove "-hpa" suffix


def get_deployment_name_from_pod_name(pod_name: str) -> str:
    return pod_name[
        : pod_name.find("-deployment")
    ]  # the pattern is <deployment_name>-deployment-<random_string>
