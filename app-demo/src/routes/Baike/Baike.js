import React from 'react';
import {NavBar, Icon,Accordion, List} from 'antd-mobile';
import knowledgeBase from '../../assets/illnessBaike.json'
import { renderRoutes } from 'react-router-config'


export default class Baike extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            activeKey:"",
            route: props.route,
            knowledgeBase:[]
        }
    }
    componentWillMount(){
        this.setState({knowledgeBase:knowledgeBase})
        if(this.props.location.state){
            this.setState({activeKey:this.props.location.state.activeKey})
        }
    }
    componentDidMount() {
        //route
        // if(this.props.location.query){
        //     this.setState({route:this.props.location.query.route})
        // }
    }
    accordChange(val){
        this.setState({activeKey:val});
        this.props.history.replace('/Baike',{activeKey:val});
    }
    goDetail(val){
        this.props.history.push({pathname:'/Baike/Detail',state:{data:val}});
    }
    goBack(){
        this.props.history.goBack()
    }
    render() {
        const route = this.state.route;
        if (this.props.location.pathname==="/Baike/Detail") {
            return(<div>
                <NavBar
                    mode="light"
                    icon={<Icon type="left" />}
                    onLeftClick={() => this.goBack()}
                    rightContent={[
                        <i className="am-icon camera" key={'camera'} onClick={()=>{this.props.history.push({pathname:'/Photo'})}}></i>
                      ]}
                    >病虫害知识库</NavBar>
                    {renderRoutes(route.children)}</div>)
        }else{
            return (
                <div>
                    <NavBar
                        mode="light"
                        icon={<Icon type="left" />}
                        onLeftClick={() => this.goBack()}
                        rightContent={[
                            <i className="am-icon camera" key={'camera'} onClick={()=>{this.props.history.push({pathname:'/Photo'})}}></i>
                          ]}
                        >病虫害知识库</NavBar>
                     <Accordion activeKey={this.state.activeKey} className="my-accordion" accordion={true} onChange={event=>this.accordChange(event)}>
                        {this.state.knowledgeBase.map((item,i)=>{
                            return <Accordion.Panel key={i} header={item.catagory}>
                                        <List className="my-list">
                                            {item.illness.map((val,index)=>{
                                                return <List.Item key={index} onClick={()=>this.goDetail(val)}>
                                                <p style={{display:"inline-block",minWidth:"0.35rem",margin:"0 0.1rem 0 0",textAlign:"right"}}>{index+1}.</p> {val.name}
                                                </List.Item>
                                            })}
                                        </List>
                                    </Accordion.Panel>
                        })}
                    </Accordion>
                </div>
            )
        }
        
    }
}