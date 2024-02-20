# kubernetes操作的库函数，最关键的是要配置好对应config文件的路径
import kubernetes as k8s
from kubernetes.client.rest import ApiException

class K8sOp:

    def __init__(self):
        # self.config = config
        # if config["is_remote"]:
        # vpc and outer net -> use .kube/config
        k8s.config.load_kube_config()

        self.k8sapi = k8s.client.CoreV1Api()
        self.k8sexapi = k8s.client.ExtensionsV1beta1Api()
        self.k8sv1_client = k8s.client.AppsV1Api()


    def get_deployment_instance_origin(self, svc_name, namespace):
        """
            给定Deployment的信息，获取具体的metadata
            if failed, return None
            raise ApiException
        """
        api_response = self.k8sv1_client.list_namespaced_deployment(namespace) # v1DeploymentList
        for item in api_response.items:
            if item.metadata.name == svc_name:
                return item
        return None

    def get_deployment_instance_num(self, svc_name, namespace):
        """
            给定Deployment的信息，获取实例数量
            if failed, return -1
            不会抛出异常
        """
        try:
            origin_response = self.get_deployment_instance_origin(svc_name, namespace)
            if origin_response == None:
                return -1
            else:
                return origin_response.spec.replicas
        except ApiException as e:
            print("Exception when calling AppsV1Api->list_namespaced_deployment: %s\n" % e)
            return -1


    def scale_deployment_by_instance(self, svc_name, namespace, instance_num_diff):
        """
            输入实例的信息，以及实例数的变动情况，进行修改
        """
        origin_response = self.get_deployment_instance_origin(svc_name, namespace)

        if origin_response == None:
            raise Exception
        elif origin_response.spec.replicas + instance_num_diff < 0:
            return -1

        origin_response.spec.replicas = int(instance_num_diff + origin_response.spec.replicas)

        api_response = self.k8sv1_client.patch_namespaced_deployment(origin_response.metadata.name,
                                                                    origin_response.metadata.namespace,
                                                                    origin_response) # v1DeploymentList
        return api_response.spec.replicas

    def scale_deployment_to_instance(self, svc_name, namespace, expected_instance_num):
        """
            输入实例的信息，以及实例数的变动情况，进行修改
        """
        origin_response = self.get_deployment_instance_origin(svc_name, namespace)
        if origin_response == None:
            raise Exception
        elif expected_instance_num < 0:
            raise Exception

        origin_response.spec.replicas = int(expected_instance_num)

        api_response = self.k8sv1_client.patch_namespaced_deployment(origin_response.metadata.name,
                                                                    origin_response.metadata.namespace,
                                                                    origin_response) # v1DeploymentList
        return api_response.spec.replicas

# op = K8sOp()
# svc_name = 'wtyfft-deploy'
# namespace='wty-istio'
# print(op.scale_deployment_to_instance(svc_name,namespace,10))