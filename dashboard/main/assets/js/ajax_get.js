export default function ajaxGet(url, successCallback, errorCallback){
  //create requeset
  var req = new XMLHttpRequest();
  req.open("GET",INVENTORY_URL);

  //handle load
  req.onload = function(){
    //if sucess, parse response text and call successCallback
    if(req.status == 200){
      console.log(req.responseText);
      console.log("Request success");

      ct = req.getResponseHeader("Content-Type");
      if(ct.indexOf('json') > -1){
        var result = JSON.parse(req.responseText);
        console.log(result);
        successCallback(result);
      }
    }
    //fail, pass error text to errorCallback
    else{
      console.log(req.responseText);
      console.log("Reqeust failed!!!");
      errorCallback(req.responseText);

    }
  }

  //handle timeout
  req.timeout = XML_TIMEOUT;
  req.ontimeout = function(){
    console.log("Request timed out");
  }

  //handle network level error
  req.onerror = function(){
    console.log("network error occured on request");
  }
  req.send();
}
