import React from 'react';

export default class Detail extends React.Component {
    constructor(props) {
        super(props)
        this.state = {
            data:{}
        }
    }
    componentDidMount() {
        // debugger
        // if(this.props.location.state){
        //     this.setState({data:this.props.location.state.detail})
        // }
        this.setState({data:this.props.location.state.data});
    }
    render() {
        return (
            <div>
                <div style={{margin:"0.3rem"}}>
                    <p>病虫害名称：{this.state.data.name}</p>
                    <p>危害病症：{this.state.data.symptom}</p>
                    <p>发病规律：{this.state.data.regularity}</p>
                    <p>防治方法：{this.state.data.methods}</p>
                    <div style={{textAlign:"center"}}>
                        <img src={require(`../../assets/${this.state.data.imgName|| 'leaf'}.jpg`) } alt="" style={{ maxHeight: "4.5rem"}}/>
                    </div>
                </div>
                
            </div>
        )
    }
}